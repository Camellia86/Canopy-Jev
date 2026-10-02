# SPDX-License-Identifier: MIT
"""JSON file/stdin client; model loading occurs only in the predict command."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from .artifacts import load_manifest


def doctor():
    packages = {}
    for name in ['torch', 'transformers', 'triton', 'flash-linear-attention', 'fla-core', 'safetensors', 'huggingface-hub', 'numpy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'packages': packages, 'python': sys.version.split()[0], 'model_loaded': False,
            'requirements': 'Linux NVIDIA CUDA, exact tested kernel sources, compatible base snapshot, >=64 GiB free GPU memory',
            'note': 'Package import/doctor do not establish CUDA execution compatibility; run a full-model parity smoke.'}


def main():
    parser = argparse.ArgumentParser(prog='canopy-jev')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    inspect = sub.add_parser('inspect'); inspect.add_argument('--artifacts', required=True, type=Path)
    predict = sub.add_parser('predict')
    predict.add_argument('--artifacts', required=True, type=Path)
    predict.add_argument('--base-model', required=True, help='Exact compatible local directory or Hugging Face repository ID')
    predict.add_argument('--revision'); predict.add_argument('--local-files-only', action='store_true')
    predict.add_argument('--request', default='-', help='JSON file, or - for stdin')
    predict.add_argument('--output', default='-', help='New output JSON file, or - for stdout')
    predict.add_argument('--device', default='cuda:0')
    predict.add_argument('--template', choices=['zero_shot_choice', 'legacy_five_demo'], default='zero_shot_choice')
    predict.add_argument('--demonstrations', type=Path)
    args = parser.parse_args()
    if args.command == 'doctor':
        result = doctor()
    elif args.command == 'inspect':
        result = load_manifest(args.artifacts)
    else:
        request = json.load(sys.stdin) if args.request == '-' else json.loads(Path(args.request).read_text())
        if args.output != '-' and Path(args.output).exists():
            parser.error('Output already exists; choose a new file')
        from .model import CanopyJev
        model = CanopyJev.from_pretrained(args.artifacts, base_model=args.base_model, revision=args.revision,
                                          device=args.device, local_files_only=args.local_files_only)
        result = model.predict(request.get('state', ''), request['questions'], template=args.template,
                               demonstrations_path=args.demonstrations)
        model.assert_unchanged()
        if args.output != '-':
            with Path(args.output).open('x') as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
            return
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
