# SPDX-License-Identifier: MIT
"""Explicit sequential GPU parity smoke against a caller-supplied reference runtime.

No benchmark labels or private examples are included. The reference module must
export Step1200Runtime; it is loaded from the supplied path, never bundled here.
Models are unloaded between runs. Both loads retain the >=64 GiB free guard.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from canopy_jev.artifacts import sha
from canopy_jev.prompt import prepare


def main():
    p = argparse.ArgumentParser(); p.add_argument('--reference-source', type=Path, required=True)
    p.add_argument('--reference-checkpoint', type=Path, required=True); p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--base-model', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true'); a = p.parse_args()
    if not a.execute:
        print(json.dumps({'plan_only': True, 'loads': 'Two sequential full-model loads; no overlap', 'required_free_gib': 64})); return
    if a.out.exists(): raise FileExistsError('Never overwrite a parity report')
    import numpy as np
    import torch
    torch.set_num_threads(6)
    sys.path.insert(0, str(a.reference_source.resolve().parent))
    spec = importlib.util.spec_from_file_location('external_canopy_reference', a.reference_source)
    reference_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference_module)
    reference = reference_module.Step1200Runtime(checkpoint=a.reference_checkpoint, min_free_gib=64.)
    state = 'A fictional service has one waiting job and its worker is available.'
    cases = [
        [{'id': 'single', 'question': 'Which action?', 'options': ['Dispatch', 'Wait']}],
        [{'id': 'left', 'question': 'Which action?', 'options': ['Dispatch', 'Wait']},
         {'id': 'right', 'question': 'Which status?', 'options': ['Ready', 'Busy', 'Unknown']}],
        [{'id': 'one', 'question': 'Select the candidate.', 'options': ['Only option']},
         {'id': 'ten', 'question': 'Which queue label?', 'options': [str(i) for i in range(10)]}]]
    expected, geometries = [], []
    for questions in cases:
        rows, root, branches = prepare(reference.tokenizer, state, questions)
        counts = [len(r['options']) for r in rows]
        value = reference.infer_tokens(root, branches, counts)
        require_repeat = reference.infer_tokens(root, branches, counts)
        if not np.array_equal(value, require_repeat): raise RuntimeError('Reference repetition is not bitexact')
        expected.append(value); geometries.append((root, branches, counts))
    reference.assert_unchanged()
    reference_identity = reference.identity
    del reference
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    from canopy_jev import CanopyJev
    model = CanopyJev.from_pretrained(a.artifacts, base_model=a.base_model, local_files_only=True)
    results = []
    for i, (root, branches, counts) in enumerate(geometries):
        actual = model.infer_tokens(root, branches, counts)
        if not np.array_equal(actual, expected[i]): raise RuntimeError('Portable loader output differs from reference at case '+str(i))
        results.append({'case': i, 'questions': len(branches), 'candidate_counts': counts,
                        'root_tokens': len(root), 'packed_tokens': len(root)+sum(map(len, branches)),
                        'float32_probabilities_sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'bitexact': True})
        # Validate complete typed JSON output too, using precisely the same prompt.
        report = model.predict(state, cases[i])
        for row, saved in zip(report['questions'], actual):
            values = list(row['probabilities'].values())
            if values != saved[:len(values)].tolist(): raise RuntimeError('JSON probability values changed')
    model.assert_unchanged()
    receipt = {'passed': True, 'synthetic_label_free_inputs': True, 'GPU_used': True,
        'reference_source_sha256': sha(a.reference_source), 'reference_checkpoint_sha256': sha(a.reference_checkpoint),
        'artifact_manifest_sha256': sha(a.artifacts/'manifest.json'), 'cases': results,
        'reference_load_seconds': reference_identity['load_seconds'], 'portable_load_seconds': model.identity['load_seconds'],
        'portable_identity': model.identity, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
        'scope': 'Small fresh full-model same-token parity smoke; no benchmark score or general kernel equivalence claim'}
    with a.out.open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps({'passed': True, 'cases': len(results), 'out': str(a.out)}))

if __name__ == '__main__': main()
