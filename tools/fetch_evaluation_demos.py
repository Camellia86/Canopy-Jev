# SPDX-License-Identifier: MIT
"""Download and reproduce the SHA-bound official MMLU-Pro validation demos.

No training or test split is downloaded. Install pyarrow==20.0.0 separately
if the optional historical evaluation template is needed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

REVISION = 'b189ec765aa7ed75c8acfea42df31fdae71f97be'
URL = 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/' + REVISION + '/data/validation-00000-of-00001.parquet'
EXPECTED = '2f161e50b47757e4b24b0b7d533d4c1cf7ca6486f664431d324dabcb0678a351'
EXPECTED_RAW = '139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078'


def fetch(out):
    if out.exists():
        raise FileExistsError('Use a new output file; official demonstrations are immutable')
    import pyarrow.parquet as pq
    raw = urllib.request.urlopen(URL, timeout=90).read()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_RAW:
        raise ValueError('Official validation source hash differs from the pinned version')
    source = pq.read_table(io.BytesIO(raw)).to_pylist()
    converted = []
    for row in source:
        options = [x for x in row['options'] if x != 'N/A']
        old_target = int(row['answer_index'])
        if row['options'][old_target] == 'N/A':
            raise ValueError('Invalid official demonstration answer')
        target = sum(x != 'N/A' for x in row['options'][:old_target])
        converted.append({'id': 'mmlu_pro:validation:' + str(row['question_id']),
            'question': row['question'], 'options': options, 'answer_index': target,
            'category': row['category'], 'cot_content': row['cot_content']})
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in converted).encode()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError('Downloaded validation projection differs from the locked protocol')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(data)
    receipt = {'repo_id': 'TIGER-Lab/MMLU-Pro', 'revision': REVISION, 'source_url': URL,
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'output_sha256': EXPECTED,
        'n': len(converted), 'train_or_test_downloaded': False, 'license': 'MIT',
        'upstream': 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro'}
    with out.with_suffix('.provenance.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/mmlu_pro_validation.jsonl'))
    args = parser.parse_args()
    print(json.dumps(fetch(args.out), indent=2))
