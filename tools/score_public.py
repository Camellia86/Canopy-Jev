# SPDX-License-Identifier: MIT
"""CPU-only aggregate scoring after complete, verified shard coverage.

Hard labels remain hard and soft target distributions remain distributions.
Unsupported/overlength rows count zero in all-planned hard accuracy; probability
metrics cover valid outputs only. No per-question error report is generated.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from canopy_jev.artifacts import require, sha, CHECKPOINT_SHA256
from evaluate_public import FAILURES, load_inputs, object_sha, read_jsonl, shard, input_shard, source_hashes, write_json, verify_prepared_binding


def validate_probability(values, k):
    import numpy as np
    require(isinstance(values, list) and len(values) == k and 1 <= k <= 10, 'Invalid candidate probability axis')
    require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in values), 'Invalid probabilities')
    require(abs(sum(values)-1.) <= 1e-6, 'Probabilities are not normalized')
    require(np.array_equal(np.asarray(values, dtype=np.float64), np.asarray(values, dtype=np.float32).astype(np.float64)), 'Probabilities are not exact saved FP32 values')


def verify_before_references(runs):
    """The only gate used by score_runs; this function never parses references."""
    runs = [Path(p) for p in runs]
    require(bool(runs) and len({p.resolve() for p in runs}) == len(runs), 'Need unique run directories')
    require(all((p/'completion.json').is_file() for p in runs), 'Every shard must complete before reference access')
    expected_sources = source_hashes()
    identities, completions, merged = [], [], []
    for run in runs:
        completion = json.loads((run/'completion.json').read_text())
        require(completion['completed'] is True and completion['references_parsed'] is False
                and completion['calibration_or_training'] is False and completion['generated_tokens'] == 0
                and completion['all_parameters_frozen_unchanged'] is True, 'Incomplete/invalid inference scope')
        require(set(completion['files_sha256']) == {'identity.json', 'plan.json', 'runtime.json', 'predictions.jsonl'}, 'Incomplete output binding')
        for name, digest in completion['files_sha256'].items():
            require(Path(name).name == name and sha(run/name) == digest, 'Changed run output: '+name)
        identity = json.loads((run/'identity.json').read_text())
        require(identity['schema'] == 1 and identity['source_checkpoint_sha256'] == CHECKPOINT_SHA256,
                'Wrong artifact/source checkpoint')
        require(identity['references_parsed'] is False and identity['candidate_truncation'] is False
                and identity['candidate_text_rewritten'] is False and identity['calibration_or_training'] is False,
                'Unsupported evaluation policy')
        require(identity['source_sha256'] == completion['source_sha256'] == expected_sources, 'Execution source identity changed')
        for name, digest in expected_sources.items():
            require(sha(run/'executed_source'/name) == digest, 'Executed source snapshot changed')
        require(all(sha(v['path']) == v['sha256'] for v in identity['files'].values()), 'Data/reference byte identity changed')
        verify_prepared_binding(identity['files']['manifest']['path'], identity['files']['inputs']['path'], identity['files']['references']['path'], identity['files'].get('unsupported', {}).get('path'))
        inputs, unsupported = load_inputs(identity['files']['inputs']['path'], identity['files'].get('unsupported', {}).get('path'))
        count, index = identity['num_shards'], identity['shard_index']
        require(type(count) is int and count >= 1 and type(index) is int and 0 <= index < count, 'Invalid shard identity')
        assigned = ([r for r in inputs if input_shard(r, count) == index]
                    + [r for r in unsupported if shard(r['id'], count) == index])
        by_input = {r['id']: r for r in assigned}
        plan = json.loads((run/'plan.json').read_text())
        require(object_sha(plan) == identity['plan_sha256'] and len(plan) == len(by_input)
                and {r['id'] for r in plan} == set(by_input), 'Planned ID coverage changed')
        predictions = read_jsonl(run/'predictions.jsonl')
        require(len(predictions) == len(plan) == completion['n_planned']
                and [r['id'] for r in predictions] == [r['id'] for r in plan], 'Incomplete/duplicate/out-of-order outputs')
        for planned, actual in zip(plan, predictions):
            required_plan = {'id', 'K', 'status', 'input_tokens'} | ({'shared_prefix_tokens'} if planned['status'] == 'pending' else set())
            require(set(planned) == required_plan, 'Unexpected/label-bearing plan fields')
            source = by_input[actual['id']]
            k = len(source['options']) if 'options' in source else source['K']
            require(planned['K'] == actual['K'] == k, 'Candidate K identity changed')
            expected_status = 'ok' if planned['status'] == 'pending' else planned['status']
            require(actual['status'] == expected_status and actual['status'] in {'ok', *FAILURES}, 'Unexpected result status')
            require({k: v for k, v in actual.items() if k not in ['status', 'probabilities']} ==
                    {k: v for k, v in planned.items() if k != 'status'}, 'Output/plan fields differ')
            require(set(actual) == set(planned)|{'probabilities'}, 'Unexpected/label-bearing output fields')
            if actual['status'] == 'ok':
                require(1 <= k <= 10 and type(actual['input_tokens']) is int
                        and actual['input_tokens'] <= identity['policy']['max_input_tokens'], 'Unsupported success row')
                validate_probability(actual['probabilities'], k)
            else:
                require(actual['probabilities'] is None, 'Never invent failure probabilities')
                if actual['status'] == 'unsupported_candidate_count':
                    require(k > 10, 'False unsupported-candidate failure')
                else:
                    require(k <= 10 and actual['input_tokens'] > identity['policy']['max_input_tokens'], 'False length failure')
        valid = sum(r['status'] == 'ok' for r in predictions)
        require(valid == completion['n_success'] == identity['statistics']['n_inference'], 'Success coverage mismatch')
        runtime = json.loads((run/'runtime.json').read_text())
        successful_plan = [p for p in plan if p['status'] == 'pending']
        timings = completion.get('batch_timings', [])
        cursor = 0
        for batch in timings:
            size = batch['rows']; require(type(size) is int and 1 <= size <= identity['policy']['max_batch'], 'Invalid batch size')
            items = successful_plan[cursor:cursor+size]; cursor += size
            require(len(items) == size and all(type(p['shared_prefix_tokens']) is int and 0 < p['shared_prefix_tokens'] < p['input_tokens'] for p in items), 'Invalid prefix/branch geometry')
            prefix = items[0]['shared_prefix_tokens']
            require(all(p['shared_prefix_tokens'] == prefix for p in items), 'Batch prefix lengths disagree')
            states = [by_input[p['id']]['state'] for p in items]
            state_texts = [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in states]
            require(len(set(state_texts)) == 1, 'A batch mixed application states')
            packed = prefix+sum(p['input_tokens']-prefix for p in items)
            require(packed == batch['packed_tokens'] and packed <= 16384
                    and (size == 1 or packed <= identity['policy']['packed_budget']), 'Batch token accounting failed')
            require(math.isfinite(batch['seconds']) and batch['seconds'] >= 0, 'Invalid batch duration')
        require(cursor == valid and len(timings) == identity['statistics']['n_packs'], 'Batch success coverage mismatch')
        if valid:
            require(runtime['source_checkpoint_sha256'] == CHECKPOINT_SHA256
                    and runtime['artifact_manifest_sha256'] == identity['artifact_manifest_sha256']
                    and runtime['parameters_frozen'] is True and runtime['cached_h60_used'] is False,
                    'Runtime/model identity mismatch')
            probe = completion['first_batch_repeat']
            require(probe['passed'] is True and probe['ids'] == [p['id'] for p in successful_plan[:timings[0]['rows']]], 'Missing/wrong first fresh batch repeat proof')
            by_pred = {r['id']: r for r in predictions}
            import numpy as np
            dense = np.zeros((len(probe['ids']), 10), dtype=np.float32)
            for i, identifier in enumerate(probe['ids']):
                require(identifier in by_pred and by_pred[identifier]['status'] == 'ok', 'Invalid repeat-proof row')
                values = by_pred[identifier]['probabilities']; dense[i, :len(values)] = values
            require(hashlib.sha256(dense.tobytes()).hexdigest() == probe['float32_probability_sha256'], 'First-batch proof differs from saved probabilities')
        else:
            require(runtime['model_loaded'] is False and completion['first_batch_repeat'] is None, 'Unexpected model run for empty supported shard')
        identities.append(identity); completions.append(completion); merged.extend(predictions)
    first = identities[0]; n_shards = first['num_shards']
    require(len(runs) == n_shards and {x['shard_index'] for x in identities} == set(range(n_shards)), 'Need every exactly-once shard before references')
    data_hashes = lambda x: {k: v['sha256'] for k, v in x['files'].items()}
    require(all(x['num_shards'] == n_shards and x['policy'] == first['policy']
                and x['artifact_manifest_sha256'] == first['artifact_manifest_sha256']
                and data_hashes(x) == data_hashes(first) for x in identities), 'Shard data/model/protocol mismatch')
    inputs, unsupported = load_inputs(first['files']['inputs']['path'], first['files'].get('unsupported', {}).get('path'))
    planned_ids = {r['id'] for r in inputs+unsupported}
    require(len(merged) == len(planned_ids) and {r['id'] for r in merged} == planned_ids, 'Merged shard population is incomplete/overlapping')
    return first, merged, [{'completion_sha256': sha(p/'completion.json'), 'shard_index': i['shard_index']} for p, i in zip(runs, identities)]


def validate_references(refs, predictions):
    require(len(refs) == len(predictions) and len({r['id'] for r in refs}) == len(refs)
            and {r['id'] for r in refs} == {r['id'] for r in predictions}, 'Complete reference population differs from planned IDs')
    by = {r['id']: r for r in predictions}
    for r in refs:
        require(all(k in r for k in ['source', 'group_id', 'split', 'family', 'K', 'target', 'is_hard', 'hard_target', 'soft_target']), 'Incomplete reference schema')
        t = r['target']; k = r['K']
        require(type(k) is int and k >= 1 and by[r['id']]['K'] == k and isinstance(t, list) and len(t) == k, 'Reference candidate axis mismatch')
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in t)
                and math.isclose(sum(t), 1., rel_tol=0, abs_tol=1e-8), 'Invalid target distribution')
        hard = sum(x == 1 for x in t) == 1 and all(x in [0, 1] for x in t)
        require(type(r['is_hard']) is bool and r['is_hard'] == hard, 'Soft/hard target semantics changed')
        require(r['hard_target'] == (t.index(1) if hard else None)
                and r['soft_target'] == (None if hard else t), 'Target was hardened or reordered')


def metric_block(refs, predictions):
    import numpy as np
    by = predictions if isinstance(predictions, dict) else {r['id']: r for r in predictions}
    hard_n = hard_correct = valid_hard = valid_soft = positive_correct = valid_n = 0
    selected_mass = nll = brier = expected_brier = score_mae = score_n = 0.
    groups = defaultdict(lambda: [0, 0, 0, 0]); statuses = Counter()
    for ref in refs:
        row = by[ref['id']]; t = np.asarray(ref['target'], dtype=np.float64); hard = ref['is_hard']
        hard_n += int(hard); g = groups[(ref['source'], ref['group_id'])]; g[0] += int(hard); g[2] += 1
        statuses[row['status']] += 1
        if row['status'] != 'ok': continue
        p = np.asarray(row['probabilities'], dtype=np.float64); chosen = int(p.argmax())
        correct = bool(hard and chosen == ref['hard_target']); positive = bool(t[chosen] > 0)
        valid_n += 1; valid_hard += int(hard); valid_soft += int(not hard); hard_correct += int(correct)
        positive_correct += int(positive); selected_mass += float(t[chosen]); g[1] += int(correct); g[3] += int(positive)
        nll += float(-(t*np.log(np.maximum(p, 1e-12))).sum())
        brier += float(((p-t)**2).sum()); expected_brier += float((p*p).sum()+1-2*(p*t).sum())
        if ref.get('kind', ref.get('type')) == 'score':
            levels = np.arange(len(p), dtype=np.float64)
            score_mae += float((t*np.abs(float(p@levels)-levels)).sum()); score_n += 1
    ratio = lambda a, b: a/b if b else None
    hard_groups = [v for v in groups.values() if v[0]]
    return {'planned_rows': len(refs), 'valid_probability_rows': valid_n, 'probability_coverage': ratio(valid_n, len(refs)),
        'status_counts': dict(statuses), 'hard_rows': hard_n, 'soft_rows': len(refs)-hard_n, 'hard_correct': hard_correct,
        'hard_accuracy_all_planned': ratio(hard_correct, hard_n), 'valid_hard_rows': valid_hard, 'valid_soft_rows': valid_soft,
        'hard_accuracy_valid_only': ratio(hard_correct, valid_hard), 'positive_set_correct': positive_correct,
        'positive_set_accuracy_all_planned': ratio(positive_correct, len(refs)), 'expected_accuracy_all_planned': ratio(selected_mass, len(refs)),
        'valid_probability_metrics': {'nll': ratio(nll, valid_n), 'brier_vs_target_distribution': ratio(brier, valid_n),
            'expected_brier': ratio(expected_brier, valid_n), 'rows': valid_n,
            'score_expected_absolute_error': ratio(score_mae, score_n), 'score_rows': int(score_n)},
        'group_macro': {'groups': len(groups), 'hard_groups': len(hard_groups),
            'soft_only_groups_excluded_from_hard': len(groups)-len(hard_groups),
            'hard_accuracy': ratio(sum(v[1]/v[0] for v in hard_groups), len(hard_groups)),
            'positive_set_accuracy': ratio(sum(v[3]/v[2] for v in groups.values()), len(groups))}}


def score_runs(runs):
    identity, predictions, receipts = verify_before_references(runs)
    # First and only reference parse, strictly after all shards/output gates.
    reference = identity['files']['references']
    require(sha(reference['path']) == reference['sha256'], 'Reference bytes changed after gate')
    refs = read_jsonl(reference['path'])
    validate_references(refs, predictions)
    by_prediction = {r['id']: r for r in predictions}
    sources, splits, groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for ref in refs:
        sources[ref['source']].append(ref); splits[ref['split']].append(ref)
        groups[(ref['source'], ref['group_id'])].append(ref)
    result = {'schema': 1, 'passed': True, 'source_checkpoint_sha256': identity['source_checkpoint_sha256'],
        'artifact_manifest_sha256': identity['artifact_manifest_sha256'], 'shard_receipts': receipts,
        'files_sha256': {k: v['sha256'] for k, v in identity['files'].items()}, 'policy': identity['policy'],
        'complete_gate_before_reference_parse': True, 'overall': metric_block(refs, by_prediction),
        'per_source': {s: metric_block(items, by_prediction) for s, items in sorted(sources.items())},
        'per_split': {s: metric_block(items, by_prediction) for s, items in sorted(splits.items())},
        'per_group': [{'source': key[0], 'group_id': key[1], **metric_block(items, by_prediction)}
                      for key, items in sorted(groups.items(), key=lambda pair: json.dumps(pair[0]))],
        'definitions': {'failures': 'Explicit K/length failures score zero for all-planned hard, positive-set and expected accuracy; no probability is invented.',
            'hard': 'Only exact one-hot targets; soft rows are never converted into hard labels.',
            'probability_metrics': 'Valid probability rows only; preserve soft target mass, denominator and coverage.',
            'group_macro': 'Equal weight per (source,group_id), not per row; soft-only groups excluded from hard group accuracy and counted.',
            'tie': 'First offered candidate at maximum probability; this public projection scorer does not substitute for JevBench231 official lexicographic-key tie handling.',
            'nll': 'Target-distribution cross-entropy using max(p,1e-12); FP64.',
            'expected_brier': 'Expected one-hot Brier under target distribution, distinct from squared distance to the target distribution.'},
        'limitations': ['Public projection denominators can differ from the original internal benchmark; consult the preparation manifest.',
            'Joint all-candidate prompt token caps differ from per-candidate sequence limits in other implementations.',
            'No per-question errors, independent model-memory attestation, clustered uncertainty interval or competitor rerun is produced.',
            'Token-count receipts and exact packing arithmetic are verified; this lightweight public scorer does not re-tokenize all inputs or replace the independent research audit.'],
        'CPU_only': True, 'scorer_sha256': sha(__file__)}
    require(sha(reference['path']) == reference['sha256'] and source_hashes() == identity['source_sha256'], 'Data/source changed during scoring')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    require(not a.out.exists(), 'Never overwrite a score report')
    report = score_runs(a.runs); write_json(a.out, report)
    print(json.dumps({'passed': True, 'planned': report['overall']['planned_rows'], 'valid': report['overall']['valid_probability_rows'], 'out': str(a.out)}))

if __name__ == '__main__': main()
