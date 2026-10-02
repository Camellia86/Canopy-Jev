# SPDX-License-Identifier: MIT
"""Tiny synthetic tests; no GPU and no actual evaluation reference files."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import evaluate_public as evaluate
import score_public as score


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(x['role']+'\n'+x['content']+'\n' for x in messages)+'assistant\n'
    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


def input_row(identifier, options=None, state='Shared synthetic state', instruction='Which?'):
    options = ['A', 'B'] if options is None else options
    return {'id': identifier, 'benchmark': 'synthetic', 'state': state, 'instructions': instruction,
            'type': 'choice', 'keys': [str(i) for i in range(len(options))], 'options': options}


def reference(identifier, target, source='s', group='g'):
    hard = target.count(1.) == 1 and all(x in [0., 1.] for x in target)
    return {'id': identifier, 'source': source, 'group_id': group, 'split': 'test', 'family': 'synthetic',
            'kind': 'score' if not hard else 'choice', 'K': len(target), 'target': target,
            'is_hard': hard, 'hard_target': target.index(1.) if hard else None, 'soft_target': None if hard else target}


class PublicTests(unittest.TestCase):
    def test_direct_options_no_duplicate_keys(self):
        row = input_row('x', ['a: First candidate', 'b: Second candidate'])
        _, packs, _, _ = evaluate.plan_inputs([row], [], Tokenizer(), {'max_batch':16, 'max_input_tokens':4096, 'packed_budget':6000})
        text = ''.join(map(chr, packs[0]['prefix']+packs[0]['branches'][0]))
        self.assertIn('A. a: First candidate', text)
        self.assertNotIn('a: a: First', text)
        self.assertNotIn('target', text)

    def test_plan_failures_and_shared_state(self):
        rows = [input_row('a'), input_row('b', state='Different state'), input_row('long', instruction='L'*4500), input_row('many', options=['x']*11)]
        unsupported = [{'id':'unsupported', 'K':12}]
        _, packs, plan, stats = evaluate.plan_inputs(rows, unsupported, Tokenizer(), {'max_batch':16, 'max_input_tokens':4096, 'packed_budget':6000})
        self.assertEqual(stats['n_planned'], 5); self.assertEqual(stats['n_inference'], 2)
        self.assertEqual(stats['failure_counts'], {'input_token_limit':1, 'unsupported_candidate_count':2})
        self.assertEqual(len(packs), 2)  # Distinct states are not put in one state group.
        self.assertEqual({r['id'] for r in plan}, {'a','b','long','many','unsupported'})

    def test_shards_exact_once(self):
        rows = [input_row(str(i)) for i in range(9)]; unsupported = [{'id':'u','K':11}]
        all_ids=[]
        for i in range(3):
            _, _, plan, _ = evaluate.plan_inputs(rows, unsupported, Tokenizer(), {'max_batch':16, 'max_input_tokens':4096, 'packed_budget':6000}, 3, i)
            all_ids += [r['id'] for r in plan]
        self.assertEqual(set(all_ids), {r['id'] for r in rows+unsupported}); self.assertEqual(len(all_ids), 10)
        self.assertEqual(len({evaluate.input_shard(r, 3) for r in rows}), 1)
        obj = input_row('json-state', state={'z':'中文','a':1})
        expected = int(hashlib.sha256(json.dumps(obj['state'], ensure_ascii=False).encode()).hexdigest(), 16) % 7
        self.assertEqual(evaluate.input_shard(obj, 7), expected)

    def test_probability_validation(self):
        score.validate_probability([.25,.75],2)
        for values in [[.2,.8],[.25,.74],[float('nan'),1.],[1.,0.,0.]]:
            with self.assertRaises(ValueError): score.validate_probability(values,2)

    def test_hard_soft_and_failure_metrics(self):
        refs=[reference('a',[1.,0.],group='g1'), reference('b',[1.,0.],group='g1'),
              reference('c',[.5,.5],group='g2'), reference('d',[0.,1.],group='g3')]
        predictions=[{'id':'a','status':'ok','K':2,'probabilities':[.75,.25]},
                     {'id':'b','status':'input_token_limit','K':2,'probabilities':None},
                     {'id':'c','status':'ok','K':2,'probabilities':[.5,.5]},
                     {'id':'d','status':'input_token_limit','K':2,'probabilities':None}]
        score.validate_references(refs,predictions)
        m=score.metric_block(refs,predictions)
        self.assertEqual(m['hard_accuracy_all_planned'],1/3); self.assertEqual(m['hard_accuracy_valid_only'],1.)
        self.assertEqual(m['valid_soft_rows'],1); self.assertEqual(m['expected_accuracy_all_planned'],.375)
        self.assertEqual(m['group_macro']['hard_accuracy'],.25)
        self.assertEqual(m['valid_probability_metrics']['brier_vs_target_distribution'],.0625)
        self.assertEqual(m['valid_probability_metrics']['expected_brier'],.3125)
        self.assertEqual(m['valid_probability_metrics']['score_expected_absolute_error'],.5)
        hardened=copy.deepcopy(refs); hardened[2]['is_hard']=True
        with self.assertRaises(ValueError):score.validate_references(hardened,predictions)

    def test_manifest_requires_unsupported_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['test_inputs.jsonl','test_references.jsonl','test_unsupported.jsonl']:
                (root/name).write_text('synthetic bytes only')
            hashes={name:evaluate.sha(root/name) for name in ['test_inputs.jsonl','test_references.jsonl','test_unsupported.jsonl']}
            (root/'manifest.json').write_text(json.dumps({'splits':{'test':{'unsupported_rows':1,'file_sha256':hashes}}}))
            with self.assertRaisesRegex(ValueError,'unsupported-ID file'):
                evaluate.verify_prepared_binding(root/'manifest.json',root/'test_inputs.jsonl',root/'test_references.jsonl')
            evaluate.verify_prepared_binding(root/'manifest.json',root/'test_inputs.jsonl',root/'test_references.jsonl',root/'test_unsupported.jsonl')

    def test_missing_completion_stops_reference_reader(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(score,'read_jsonl') as reader:
            with self.assertRaisesRegex(ValueError,'Every shard must complete'):
                score.score_runs([Path(temp)])
            reader.assert_not_called()

    def test_complete_synthetic_shards_and_tamper_gate(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); inputs=[input_row('valid'), input_row('soft')]
            unsupported=[{'id':'many','K':11,'reason':'Synthetic K limit'}]
            refs=[reference('valid',[1.,0.]),reference('soft',[.5,.5]),reference('many',[1.]+[0.]*10)]
            for name, rows in [('inputs',inputs),('references',refs),('unsupported',unsupported)]:
                (root/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (root/'manifest.json').write_text(json.dumps({'inputs_sha256':evaluate.sha(root/'inputs.jsonl'),'references_sha256':evaluate.sha(root/'references.jsonl'),'unsupported_sha256':evaluate.sha(root/'unsupported.jsonl')}))
            files={k:{'path':str(root/(k+'.jsonl' if k!='manifest' else k+'.json')),
                       'sha256':evaluate.sha(root/(k+'.jsonl' if k!='manifest' else k+'.json'))}
                   for k in ['inputs','references','unsupported','manifest']}
            sources=evaluate.source_hashes(); runs=[]
            for index in range(2):
                run=root/('run'+str(index));run.mkdir();runs.append(run)
                selected,packs,plan,stats=evaluate.plan_inputs(inputs,unsupported,Tokenizer(),{'max_batch':16,'max_input_tokens':4096,'packed_budget':6000},2,index)
                identity={'schema':1,'source_checkpoint_sha256':score.CHECKPOINT_SHA256,'artifact_manifest_sha256':'synthetic',
                          'files':files,'num_shards':2,'shard_index':index,'policy':{'max_batch':16,'max_input_tokens':4096,'packed_budget':6000},
                          'source_sha256':sources,'statistics':stats,'plan_sha256':evaluate.object_sha(plan),
                          'references_parsed':False,'candidate_truncation':False,'candidate_text_rewritten':False,'calibration_or_training':False}
                evaluate.write_json(run/'identity.json',identity);evaluate.write_json(run/'plan.json',plan)
                runtime={'source_checkpoint_sha256':score.CHECKPOINT_SHA256,'artifact_manifest_sha256':'synthetic',
                         'parameters_frozen':True,'cached_h60_used':False} if packs else {'model_loaded':False}
                evaluate.write_json(run/'runtime.json',runtime)
                outputs=[{**p,'status':'ok' if p['status']=='pending' else p['status'],
                          'probabilities':[.75,.25] if p['status']=='pending' else None} for p in plan]
                (run/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
                probe=None
                if packs:
                    ids=[selected[i]['id'] for i in packs[0]['indices']]
                    dense=np.zeros((len(ids),10),dtype=np.float32);dense[:,:2]=[.75,.25]
                    probe={'passed':True,'ids':ids,'float32_probability_sha256':hashlib.sha256(dense.tobytes()).hexdigest()}
                for name in sources:
                    target=run/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(evaluate.ROOT/name,target)
                completion={'completed':True,'references_parsed':False,'calibration_or_training':False,'generated_tokens':0,
                            'all_parameters_frozen_unchanged':True,'source_sha256':sources,'n_planned':len(plan),'n_success':stats['n_inference'],
                            'first_batch_repeat':probe,'batch_timings':[{'rows':len(p['indices']),'packed_tokens':p['packed_tokens'],'seconds':.1} for p in packs],'files_sha256':{p.name:evaluate.sha(p) for p in run.iterdir() if p.is_file()}}
                evaluate.write_json(run/'completion.json',completion)
            result=score.score_runs(runs)
            self.assertTrue(result['complete_gate_before_reference_parse']);self.assertEqual(result['overall']['planned_rows'],3)
            self.assertEqual(result['overall']['hard_accuracy_all_planned'],.5)
            self.assertEqual(result['overall']['hard_accuracy_valid_only'],1.)
            self.assertEqual(result['overall']['soft_rows'],1)
            with patch.object(score,'read_jsonl', wraps=score.read_jsonl) as reader:
                with self.assertRaisesRegex(ValueError,'exactly-once shard'):score.score_runs(runs[:1])
                self.assertFalse(any(str(c.args[0]).endswith('references.jsonl') for c in reader.call_args_list))
            (runs[0]/'predictions.jsonl').write_text('')
            with patch.object(score,'read_jsonl') as reader:
                with self.assertRaisesRegex(ValueError,'Changed run output'):score.score_runs(runs)
                reader.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
