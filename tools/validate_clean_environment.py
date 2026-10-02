# SPDX-License-Identifier: MIT
"""Explicit full-model check against the published label-free parity hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'plan_only': True, 'model_loaded': False, 'minimum_free_gib': 64}))
        return
    if args.out.exists():
        raise FileExistsError('Do not overwrite a validation receipt')
    from canopy_jev import CanopyJev
    from canopy_jev.prompt import prepare
    import torch
    torch.set_num_threads(6)
    expected = json.loads(args.expected.read_text())
    assert expected['passed'] and expected['synthetic_label_free_inputs']
    model = CanopyJev.from_pretrained(args.artifacts, base_model=args.base_model, local_files_only=True)
    state = 'A fictional service has one waiting job and its worker is available.'
    cases = [
        [{'id': 'single', 'question': 'Which action?', 'options': ['Dispatch', 'Wait']}],
        [{'id': 'left', 'question': 'Which action?', 'options': ['Dispatch', 'Wait']},
         {'id': 'right', 'question': 'Which status?', 'options': ['Ready', 'Busy', 'Unknown']}],
        [{'id': 'one', 'question': 'Select the candidate.', 'options': ['Only option']},
         {'id': 'ten', 'question': 'Which queue label?', 'options': [str(i) for i in range(10)]}]]
    evidence = []
    for index, questions in enumerate(cases):
        rows, prefix, branches = prepare(model.tokenizer, state, questions)
        counts = [len(row['options']) for row in rows]
        p = model.infer_tokens(prefix, branches, counts)
        digest = hashlib.sha256(p.tobytes()).hexdigest()
        assert digest == expected['cases'][index]['float32_probabilities_sha256']
        result = model.predict(state, questions)
        assert all(list(row['probabilities'].values()) == values[:len(row['probabilities'])].tolist()
            for row, values in zip(result['questions'], p))
        evidence.append({'case': index, 'float32_probabilities_sha256': digest, 'bitexact': True})
    model.assert_unchanged()
    receipt = {'passed': True, 'GPU_used': True, 'model': 'Canopy-Jev',
        'artifact_manifest_sha256': hashlib.sha256((args.artifacts / 'manifest.json').read_bytes()).hexdigest(),
        'cases': evidence, 'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'load_seconds': model.identity['load_seconds'], 'all_parameters_frozen_unchanged': True,
        'expected_receipt_sha256': hashlib.sha256(args.expected.read_bytes()).hexdigest(),
        'scope': 'Actual clean-environment full-model small same-token smoke, not a new benchmark or latency study',
        'recorded_unix_time': time.time()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
