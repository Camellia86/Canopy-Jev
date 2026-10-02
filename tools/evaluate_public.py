# SPDX-License-Identifier: MIT
"""Portable reference-free inference for prepared public evaluation inputs.

Target/reference files are hashed but never parsed. Every planned ID receives
an output, including explicit K/length failures; fatal model errors leave no
completion receipt. No calibration, training, candidate truncation or retry.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from canopy_jev.artifacts import load_manifest, require, sha
from canopy_jev.prompt import common_prefix, zero_shot_prompt

INPUT_FIELDS = {'id', 'benchmark', 'state', 'instructions', 'type', 'keys', 'options'}
FAILURES = {'unsupported_candidate_count', 'input_token_limit'}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(s) for s in stream if s.strip()]


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    require((ROOT/'src/canopy_jev/__init__.py').is_file(), 'Public evaluation tools require the complete release checkout, including src/')
    files = [ROOT/'tools/evaluate_public.py', ROOT/'tools/score_public.py']
    files += sorted((ROOT/'src/canopy_jev').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def shard(identifier, count):
    return int(hashlib.sha256(identifier.encode()).hexdigest(), 16) % count


def input_shard(row, count):
    state = row['state'] if isinstance(row['state'], str) else json.dumps(row['state'], ensure_ascii=False)
    return int(hashlib.sha256(state.encode()).hexdigest(), 16) % count


def load_inputs(inputs_path, unsupported_path=None):
    inputs = read_jsonl(inputs_path)
    unsupported = read_jsonl(unsupported_path) if unsupported_path is not None else []
    require(inputs or unsupported, 'Empty planned population')
    for r in inputs:
        require(set(r) == INPUT_FIELDS, 'Input fields must be exactly the reference-free schema')
        require(isinstance(r['id'], str) and r['id'] and isinstance(r['instructions'], str), 'Invalid input identity/instructions')
        require(r['type'] in ['choice', 'noul', 'score'], 'Unknown decision kind')
        require(isinstance(r['options'], list) and len(r['options']) >= 1
                and all(isinstance(v, str) for v in r['options']), 'Invalid options')
        require(isinstance(r['keys'], list) and len(r['keys']) == len(r['options'])
                and len(set(r['keys'])) == len(r['keys']) and all(isinstance(v, str) for v in r['keys']), 'Invalid ordered candidate keys')
        require(r['type'] != 'noul' or len(r['keys']) == 2, 'Noul must retain its original two-option axis')
    for r in unsupported:
        require(isinstance(r['id'], str) and type(r['K']) is int and r['K'] > 10, 'Unsupported file must record original K>10 IDs')
        require(not any(k in r for k in ['target', 'answer', 'answer_index', 'hard_target', 'soft_target', 'reference', 'rationale']), 'Unsupported records must be label-free')
    ids = [r['id'] for r in inputs+unsupported]
    require(len(ids) == len(set(ids)), 'Duplicate or overlapping planned IDs')
    return inputs, unsupported


def verify_prepared_binding(manifest_path, inputs_path, references_path, unsupported_path=None):
    """Check prepared-file identities without parsing references."""
    manifest = json.loads(Path(manifest_path).read_text())
    actual_inputs, actual_references = sha(inputs_path), sha(references_path)
    if 'inputs_sha256' in manifest:
        require(manifest['inputs_sha256'] == actual_inputs and manifest['references_sha256'] == actual_references,
                'Prepared manifest input/reference checksum mismatch')
        if unsupported_path is not None:
            require(manifest.get('unsupported_sha256') == sha(unsupported_path), 'Unbound unsupported-ID file')
        else:
            require(not manifest.get('unsupported_sha256'), 'Prepared unsupported IDs must be included')
    else:
        matched = []
        for entry in manifest.get('splits', {}).values():
            hashes = entry.get('file_sha256', {})
            if hashes.get(Path(inputs_path).name) == actual_inputs and hashes.get(Path(references_path).name) == actual_references:
                matched.append(entry)
        require(len(matched) == 1, 'Inputs and references must belong to exactly one prepared split')
        entry = matched[0]
        if unsupported_path is not None:
            require(entry['file_sha256'].get(Path(unsupported_path).name) == sha(unsupported_path), 'Prepared unsupported-ID checksum mismatch')
        else:
            require(entry.get('unsupported_rows', 0) == 0, 'Include the prepared unsupported-ID file; do not shrink the planned denominator')
    return manifest


def plan_inputs(inputs, unsupported, tokenizer, policy, num_shards=1, shard_index=0):
    require(type(num_shards) is int and num_shards >= 1 and type(shard_index) is int and 0 <= shard_index < num_shards, 'Invalid shard')
    require(type(policy['max_batch']) is int and 1 <= policy['max_batch'] <= 16, 'max_batch must be 1..16')
    require(type(policy['packed_budget']) is int and 1 <= policy['packed_budget'] <= 16384, 'Invalid packed-token budget')
    require(type(policy['max_input_tokens']) is int and 1 <= policy['max_input_tokens'] <= 16384, 'Invalid input-token limit')
    selected = [r for r in inputs if input_shard(r, num_shards) == shard_index]
    unsupported = [r for r in unsupported if shard(r['id'], num_shards) == shard_index]
    prompts, failures, groups = {}, [], defaultdict(list)
    for i, row in enumerate(selected):
        k = len(row['options'])
        if k > 10:
            failures.append({'id': row['id'], 'K': k, 'status': 'unsupported_candidate_count', 'input_tokens': None}); continue
        # Direct option strings: no second criteria/key prefix is injected.
        prompt = zero_shot_prompt(tokenizer, row['state'], row)
        prompts[i] = prompt
        if len(prompt[0]) > policy['max_input_tokens']:
            failures.append({'id': row['id'], 'K': k, 'status': 'input_token_limit', 'input_tokens': len(prompt[0])}); continue
        state = row['state'] if isinstance(row['state'], str) else json.dumps(row['state'], ensure_ascii=False)
        groups[hashlib.sha256(state.encode()).hexdigest()].append(i)
    packs, plan = [], []
    for _, indices in sorted(groups.items()):
        pending = sorted(indices, key=lambda i: (len(prompts[i][0]), i))
        while pending:
            batch = [pending[0]]; prefix = common_prefix([prompts[pending[0]]])
            for i in pending[1:policy['max_batch']]:
                trial = batch+[i]; pl = common_prefix([prompts[j] for j in trial])
                if pl+sum(len(prompts[j][0])-pl for j in trial) > policy['packed_budget']: break
                batch, prefix = trial, pl
            pending = pending[len(batch):]
            branches = [prompts[i][0][prefix:] for i in batch]
            packed = prefix+sum(map(len, branches))
            require(prefix > 0 and all(branches) and packed <= 16384, 'Invalid complete tree geometry')
            require(len(batch) == 1 or packed <= policy['packed_budget'], 'Multirow tree exceeds budget')
            packs.append({'indices': batch, 'prefix': prompts[batch[0]][0][:prefix], 'branches': branches,
                          'candidate_counts': [len(selected[i]['options']) for i in batch], 'packed_tokens': packed})
            plan += [{'id': selected[i]['id'], 'K': len(selected[i]['options']), 'status': 'pending',
                      'input_tokens': len(prompts[i][0]), 'shared_prefix_tokens': prefix} for i in batch]
    failures += [{'id': r['id'], 'K': r['K'], 'status': 'unsupported_candidate_count', 'input_tokens': None} for r in unsupported]
    plan += failures
    expected = {r['id'] for r in selected+unsupported}
    require(len(plan) == len(expected) and {r['id'] for r in plan} == expected, 'Plan coverage failure')
    stats = {'n_planned': len(plan), 'n_inference': sum(len(p['indices']) for p in packs), 'n_packs': len(packs),
             'failure_counts': dict(Counter(p['status'] for p in failures)),
             'max_input_tokens_observed': max((len(p[0]) for p in prompts.values()), default=0),
             'max_packed_tokens': max((p['packed_tokens'] for p in packs), default=0)}
    return selected, packs, plan, stats


def run_evaluation(*, inputs_path, references_path, manifest_path, unsupported_path, artifact_dir,
                   base_model, revision, local_files_only, out, num_shards, shard_index, policy,
                   preflight=False, device='cuda:0'):
    out = Path(out)
    require(preflight or not out.exists(), 'Never overwrite a run directory')
    artifact = load_manifest(artifact_dir)
    files = {k: {'path': str(Path(p).resolve()), 'sha256': sha(p)} for k, p in
             [('inputs', inputs_path), ('references', references_path), ('manifest', manifest_path)]}
    if unsupported_path is not None:
        files['unsupported'] = {'path': str(Path(unsupported_path).resolve()), 'sha256': sha(unsupported_path)}
    verify_prepared_binding(manifest_path, inputs_path, references_path, unsupported_path)
    inputs, unsupported = load_inputs(inputs_path, unsupported_path)
    from canopy_jev.artifacts import resolve_base_model, verify_base_model
    from transformers import AutoTokenizer
    base = resolve_base_model(base_model, revision, local_files_only)
    verify_base_model(base, artifact)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    selected, packs, plan, stats = plan_inputs(inputs, unsupported, tokenizer, policy, num_shards, shard_index)
    if preflight:
        return {'preflight': True, 'model_loaded': False, 'references_parsed': False, **stats}
    sources = source_hashes()
    identity = {'schema': 1, 'purpose': 'All-planned public evaluation; no fitting or test-driven model selection',
                'source_checkpoint_sha256': artifact['source_checkpoint_sha256'],
                'artifact_manifest_sha256': sha(Path(artifact_dir)/'manifest.json'), 'files': files,
                'num_shards': num_shards, 'shard_index': shard_index,
                'shard_scheme': 'supported: SHA256(exact serialized state) modulo N; unsupported: SHA256(id) modulo N', 'policy': policy,
                'source_sha256': sources, 'statistics': stats, 'plan_sha256': object_sha(plan),
                'prompt': 'Exact zero-shot Application state / Question / Options adapter',
                'candidate_text_rewritten': False, 'candidate_truncation': False,
                'references_parsed': False, 'generated_tokens': 0, 'calibration_or_training': False}
    out.mkdir(parents=True)
    for name in sources:
        target = out/'executed_source'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    write_json(out/'identity.json', identity); write_json(out/'plan.json', plan)
    runtime = None
    if packs:
        from canopy_jev import CanopyJev
        runtime = CanopyJev.from_pretrained(artifact_dir, base_model=str(base), local_files_only=True,
                                           min_free_gib=64., max_tokens=16384, device=device)
    runtime_identity = runtime.identity if runtime else {'model_loaded': False, 'reason': 'No supported, in-limit requests assigned to this shard'}
    write_json(out/'runtime.json', runtime_identity)
    pending_plan = {r['id']: r for r in plan}; count = 0; timings = []; first_repeat = None
    with (out/'predictions.jsonl').open('x') as stream:
        for pack in packs:
            import torch
            import numpy as np
            torch.cuda.synchronize(runtime.device); tick = time.perf_counter()
            probability = runtime.infer_tokens(pack['prefix'], pack['branches'], pack['candidate_counts'])
            torch.cuda.synchronize(runtime.device); duration = time.perf_counter()-tick
            if first_repeat is None:
                repeated = runtime.infer_tokens(pack['prefix'], pack['branches'], pack['candidate_counts'])
                require(np.array_equal(probability, repeated), 'First fresh batch is not repeatable')
                first_repeat = {'passed': True, 'ids': [selected[i]['id'] for i in pack['indices']],
                                'float32_probability_sha256': hashlib.sha256(probability.tobytes()).hexdigest()}
            timings.append({'rows': len(pack['indices']), 'packed_tokens': pack['packed_tokens'], 'seconds': duration})
            for slot, i in enumerate(pack['indices']):
                original = selected[i]; planned = pending_plan[original['id']]
                values = probability[slot, :planned['K']].tolist()
                row = {**planned, 'status': 'ok', 'probabilities': values}
                stream.write(json.dumps(row, allow_nan=False)+'\n'); count += 1
            stream.flush()
            print(json.dumps({'inference_rows_completed': count, 'planned_shard': len(plan)}), flush=True)
        for planned in plan:
            if planned['status'] in FAILURES:
                stream.write(json.dumps({**planned, 'probabilities': None}, allow_nan=False)+'\n'); count += 1
        stream.flush()
    if runtime is not None: runtime.assert_unchanged()
    require(count == len(plan), 'Run omitted planned outputs')
    require(sources == source_hashes(), 'Execution sources changed')
    require(all(sha(entry['path']) == entry['sha256'] for entry in files.values()), 'Inputs or reference bytes changed')
    completion = {'completed': True, 'n_planned': count, 'n_success': stats['n_inference'],
                  'references_parsed': False, 'generated_tokens': 0, 'calibration_or_training': False,
                  'all_parameters_frozen_unchanged': True, 'source_sha256': sources, 'first_batch_repeat': first_repeat,
                  'batch_timings': timings, 'files_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'completion.json', completion)
    return {'completed': True, **stats, 'references_parsed': False, 'out': str(out)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--references', type=Path, required=True, help='Hash only; never parsed by inference')
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--unsupported', type=Path)
    p.add_argument('--artifacts', type=Path, required=True); p.add_argument('--base-model', required=True); p.add_argument('--revision')
    p.add_argument('--local-files-only', action='store_true'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-shards', type=int, default=1); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--max-input-tokens', type=int, default=4096); p.add_argument('--packed-budget', type=int, default=6000)
    p.add_argument('--max-batch', type=int, default=16); p.add_argument('--preflight', action='store_true'); p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    result = run_evaluation(inputs_path=a.inputs, references_path=a.references, manifest_path=a.manifest,
                           unsupported_path=a.unsupported, artifact_dir=a.artifacts, base_model=a.base_model,
                           revision=a.revision, local_files_only=a.local_files_only, out=a.out,
                           num_shards=a.num_shards, shard_index=a.shard_index,
                           policy={'max_input_tokens': a.max_input_tokens, 'packed_budget': a.packed_budget, 'max_batch': a.max_batch},
                           preflight=a.preflight, device=a.device)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
