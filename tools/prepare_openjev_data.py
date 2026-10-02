# SPDX-License-Identifier: MIT
"""CPU-only pinned evaluation-data preparation, with no model or training calls.

Downloads only JF100 public evaluation material and/or the two public Open-Jev
v1.1 evaluation splits. Targets remain in separate reference files. Candidates
are never truncated: K>10 rows are explicitly recorded as unsupported.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = '10ad6888333fa97f8c948192797bad3de3040802'
HF_ROOT = 'https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1/resolve/' + HF_REVISION + '/'
JF_REVISION = '9abacec47394f3b393f81fbe3cdd524f028bc088'
JF_ROOT = 'https://raw.githubusercontent.com/softpudding/jev-frontier-100/' + JF_REVISION + '/'
EXPORT_SHA = 'a335f044b814ce9c2b33dc8c63ed2feab82681155db7c3ac58ecce8b87fdd615'
JF_FILES = {
    'data/items.jsonl': 'dd107ba90de381eaa479408492e81d7222115782e5fe601ab43986f94cd1d0fa',
    'data/manifest.json': '36b770066f7f3f0b17321a41b09699d5101c1b7b3db889fccab5f457fb035060',
    'docs/PROTOCOL.md': 'abafc61540543101fdd49593a20245f1d2522bd60f84d45402bf27ee0532a842',
    'src/jf100/core.py': 'df6f2eeb5da5cfca5d0f180d5dab861f4d8f5526057a391490e55c1159a28b7e',
    'LICENSE': None,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def download(url, path, expected=None):
    path = Path(path)
    require(not path.exists(), 'Refuse to overwrite downloaded evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as remote, path.open('xb') as dest:
        while True:
            block = remote.read(1024**2)
            if not block:
                break
            dest.write(block)
    digest = sha(path)
    require(expected is None or digest == expected, 'Downloaded bytes mismatch: '+url)
    return {'url': url, 'sha256': digest, 'bytes': path.stat().st_size}


def jf100(out):
    require(not out.exists(), 'Never overwrite a prepared suite')
    out.mkdir(parents=True)
    evidence = {name: download(JF_ROOT+name, out/'upstream'/name, digest) for name, digest in JF_FILES.items()}
    items = [json.loads(x) for x in (out/'upstream/data/items.jsonl').read_text().splitlines() if x.strip()]
    require(len(items) == 100 and len({r['id'] for r in items}) == 100, 'JF100 item coverage')
    pairs = defaultdict(list)
    for item in items:
        pairs[item['pair_id']].append(item)
    require(len(pairs) == 50 and all(len(v) == 2 and len({x['domain'] for x in v}) == 1 for v in pairs.values()),
            'JF100 counterfactual pair geometry')
    domains = Counter()
    with (out/'inputs.jsonl').open('x') as inputs, (out/'references.jsonl').open('x') as refs:
        for item in items:
            require(list(item['options']) == list('ABCD') and len(set(item['options'].values())) == 4
                    and item['answer'] in list('ABCD'), 'JF100 original candidate axis')
            original = list(item['options'].items())
            for trial in range(3):
                rotated = original[trial:] + original[:trial]
                keys = list('ABCD')
                target_index = (keys.index(item['answer'])-trial) % 4
                target = [float(i == target_index) for i in range(4)]
                identifier = 'jf100/'+item['id']+'/rotation_'+str(trial)
                # Only the public question/state/options enter this file.
                line(inputs, {'id': identifier, 'benchmark': 'jf100', 'type': 'choice',
                    'state': item['state'], 'instructions': item['question'], 'keys': keys,
                    'options': [key+': '+text for key, (_, text) in zip(keys, rotated)]})
                line(refs, {'id': identifier, 'item_id': item['id'], 'pair_id': item['pair_id'],
                    'trial': trial, 'trial_seed_label': [101, 202, 303][trial], 'domain': item['domain'],
                    'difficulty': item['difficulty'], 'target': target, 'target_key': keys[target_index],
                    'source': 'jf100', 'group_id': item['pair_id'], 'split': 'test', 'family': item['domain'],
                    'kind': 'choice', 'K': 4, 'is_hard': True, 'hard_target': target_index, 'soft_target': None})
                domains[item['domain']] += 1
    manifest = {'schema': 1, 'suite': 'JF100 v0.1.0 items / v0.2-budget protocol',
        'source_revision': JF_REVISION, 'n_items': 100, 'n_pairs': 50, 'n_decisions': 300,
        'K': 4, 'domain_decisions': dict(domains), 'source_files': evidence,
        'inputs_sha256': sha(out/'inputs.jsonl'), 'references_sha256': sha(out/'references.jsonl'),
        'selection': 'all 100 items, rotations 0/1/2; no sampling, no outcome-driven choices',
        'uncertainty_unit': '50 counterfactual template pairs; not 300 independent examples',
        'recommended_comparison_bootstrap': {'draws': 5000, 'seed': 92026, 'stratification': 'domain', 'unit': 'pair'},
        'published_checkpoint_scope': 'Open-Jev archived 100-step pilots only; final 27B-v1.1 JF100 is pending',
        'model_input_reference_free': True, 'competitor_responses_downloaded': False,
        'license': 'MIT; exact upstream LICENSE retained', 'inference_started': False,
        'preparer_sha256': sha(__file__)}
    write_json(out/'manifest.json', manifest)
    return manifest


def public_projection(out):
    require(not out.exists(), 'Never overwrite a prepared suite')
    out.mkdir(parents=True)
    evidence = {'export-manifest.json': download(HF_ROOT+'export-manifest.json', out/'upstream/export-manifest.json', EXPORT_SHA)}
    export = json.loads((out/'upstream/export-manifest.json').read_text())
    require(export['repo_id'] == 'ZefanCai/Open-Jev-v1.1' and export['count'] == 326619, 'Wrong public projection')
    for name in ['README.md', 'LICENSE-DATA.md', 'LICENSE-CODE-MIT', 'THIRD_PARTY_NOTICES.md', 'provenance/WANLI-README.md']:
        evidence[name] = download(HF_ROOT+name, out/'upstream'/name, export['published_files'][name]['sha256'])
    stats = {}
    all_ids, group_splits = set(), {}
    for split, expected_count in [('test', 43125), ('ood', 84267)]:
        raw_name = 'raw/community-hard-mix-v2-redistributable/'+split+'.jsonl.gz'
        raw_path = out/'upstream'/raw_name
        evidence[raw_name] = download(HF_ROOT+raw_name, raw_path, export['published_files'][raw_name]['sha256'])
        total = supported = hard = soft = supported_hard = supported_soft = 0
        by_k, by_source, by_kind, unsupported_sources = Counter(), Counter(), Counter(), Counter()
        source_details = defaultdict(lambda: Counter())
        with gzip.open(raw_path, 'rt', encoding='utf-8') as raw, (out/(split+'_inputs.jsonl')).open('x') as inputs, \
                (out/(split+'_references.jsonl')).open('x') as refs, (out/(split+'_unsupported.jsonl')).open('x') as unsupported:
            for raw_line in raw:
                if not raw_line.strip():
                    continue
                r = json.loads(raw_line)
                require(r['split'] == split and r['id'] not in all_ids, 'Duplicate/wrong-split input record')
                all_ids.add(r['id'])
                group = (r['source'], r['group_id'])
                require(group not in group_splits or group_splits[group] == split, 'Source group crosses test/OOD')
                group_splits[group] = split
                require(r['source'] != 'wikispeedia-v1', 'Unexpected restricted Wiki row')
                options, target = r['options'], r['target']
                k = len(options)
                require(k >= 1 and all(isinstance(x, str) for x in options) and len(target) == k,
                        'Invalid original candidate axis')
                require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) and 0 <= v <= 1 for v in target)
                        and math.isclose(sum(target), 1., abs_tol=1e-8, rel_tol=0), 'Invalid target distribution')
                require(r['kind'] in ['choice', 'noul', 'score'], 'Unknown typed decision kind')
                is_hard = sum(v == 1 for v in target) == 1 and all(v in [0, 1] for v in target)
                total += 1
                hard += int(is_hard)
                soft += int(not is_hard)
                by_k[k] += 1
                by_source[r['source']] += 1
                by_kind[r['kind']] += 1
                source_details[r['source']].update({'n': 1, 'hard': int(is_hard), 'soft': int(not is_hard)})
                identifier = 'openjev_v1_1/'+r['id']
                line(refs, {'id': identifier, 'original_id': r['id'], 'split': split, 'source': r['source'],
                    'group_id': r['group_id'], 'family': r.get('metadata', {}).get('family', r['source']),
                    'kind': r['kind'], 'K': k, 'target': target, 'is_hard': is_hard,
                    'hard_target': target.index(1) if is_hard else None,
                    'soft_target': None if is_hard else target,
                    'target_basis': r.get('metadata', {}).get('target_basis')})
                if k > 10:
                    unsupported_sources[r['source']] += 1
                    source_details[r['source']]['unsupported_K'] += 1
                    line(unsupported, {'id': identifier, 'original_id': r['id'], 'source': r['source'], 'group_id': r['group_id'],
                        'split': split, 'kind': r['kind'], 'K': k, 'reason': 'locked head supports at most 10 candidates; no truncation'})
                    continue
                keys = ['no', 'yes'] if r['kind'] == 'noul' else [str(i) for i in range(k)]
                require(len(keys) == k, 'Noul must preserve original two-option axis')
                line(inputs, {'id': identifier, 'benchmark': 'openjev_public_v1_1', 'type': r['kind'],
                    'state': r['state'], 'instructions': r['question'], 'keys': keys, 'options': options})
                supported += 1
                supported_hard += int(is_hard)
                supported_soft += int(not is_hard)
        require(total == expected_count == export['splits'][split]['count'], 'Incomplete public split')
        stats[split] = {'public_rows': total, 'supported_rows': supported, 'unsupported_rows': total-supported,
            'hard_rows': hard, 'soft_rows': soft, 'supported_hard_rows': supported_hard,
            'supported_soft_rows': supported_soft, 'K_counts': dict(sorted(by_k.items())),
            'by_kind': dict(by_kind), 'by_source': dict(by_source),
            'source_details': {s: dict(v) for s, v in source_details.items()},
            'unsupported_sources': dict(unsupported_sources), 'wiki_rows_missing_from_original': export['splits'][split]['excluded_count'],
            'file_sha256': {name: sha(out/name) for name in [split+'_inputs.jsonl', split+'_references.jsonl', split+'_unsupported.jsonl']}}
    manifest = {'schema': 1, 'suite': 'Open-Jev-v1.1 redistributable test/OOD projection',
        'hf_revision': HF_REVISION, 'source_files': evidence, 'splits': stats,
        'public_total': sum(s['public_rows'] for s in stats.values()),
        'supported_total': sum(s['supported_rows'] for s in stats.values()),
        'head_candidate_limit': 10, 'candidate_truncation_performed': False,
        'sampling_performed': False, 'selection': 'All original public test/OOD rows with K<=10; explicit unsupported-ID files retain the rest.',
        'full_original_internal_metric_directly_comparable': False,
        'unavailable_original_wiki_rows': 395,
        'model_input_reference_free': True, 'train_or_calibration_downloaded': False,
        'reference_coverage': 'Reference files contain every public row, including K>10; input files contain only the explicitly supported K<=10 rows.',
        'length_status': 'Not tokenized; evaluator must preflight all frozen inputs and report unsupported lengths without truncation.',
        'hard_accuracy': 'Only exact one-hot target rows; soft targets excluded and counted.',
        'probability_metrics': 'All valid supported rows; retain target distributions, never turn soft targets into hard labels.',
        'license': 'WANLI CC BY 4.0; original generated records CC0; TypeSafe wording qualification retained in upstream notices.',
        'inference_started': False, 'preparer_sha256': sha(__file__)}
    write_json(out/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['jf100', 'public-v1.1', 'all'], default='jf100')
    parser.add_argument('--out', type=Path, default=ROOT/'data/openjev_extended_eval_v1')
    args = parser.parse_args()
    result = {}
    if args.suite in ['jf100', 'all']:
        m = jf100(args.out/'jf100')
        result['jf100'] = {'decisions': m['n_decisions'], 'pairs': m['n_pairs']}
    if args.suite in ['public-v1.1', 'all']:
        m = public_projection(args.out/'public_v1_1')
        result['public_v1_1'] = {s: {k: v[k] for k in ['public_rows', 'supported_rows', 'unsupported_rows', 'hard_rows', 'soft_rows']}
                                   for s, v in m['splits'].items()}
    print(json.dumps({'passed': True, 'suites': result, 'inference_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
