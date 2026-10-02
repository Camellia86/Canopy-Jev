# SPDX-License-Identifier: MIT
"""CPU-only, label-free conversion of the approved weights to safe tensors."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from canopy_jev.artifacts import CHECKPOINT_SHA256, PRIOR_SHA256, require, sha, state_sha


def export(checkpoint, prior_path, base_model, out):
    import torch
    from safetensors.torch import save_file
    require(not torch.cuda.is_initialized(), 'CPU-only export')
    require(sha(checkpoint) == CHECKPOINT_SHA256 and sha(prior_path) == PRIOR_SHA256, 'Wrong source artifact')
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    prior = torch.load(prior_path, map_location='cpu', weights_only=True)
    require(cp['arm'] == 'rl64_weight01' and cp['step'] == 1200, 'Wrong continuation checkpoint')
    require(cp['prior_checkpoint_sha256'] == PRIOR_SHA256, 'Wrong source prior binding')
    require(cp['policy_exploration'] == .01 and cp['temperature'] == 1.1422139406204224, 'Wrong source policy')
    base = Path(base_model); head = cp['head_identity']
    require(sha(base/'config.json') == head['config_sha256'] and sha(base/'model.safetensors.index.json') == head['index_sha256'], 'Wrong base identity')
    adapters = {k: v.detach().cpu().contiguous() for k, v in cp['adapters'].items()}
    state = {k: v.detach().cpu().contiguous() for k, v in prior['state_dict'].items()}
    require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in [*adapters.values(), *state.values()]), 'Weights must preserve finite FP32 values')
    require(set(state) == {'weight', 'bias', 'log_temperature'}, 'Unexpected prior state')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    require(not any((out/name).exists() for name in ['manifest.json', 'adapter.safetensors', 'prior.safetensors']), 'Do not overwrite an export')
    save_file(adapters, str(out/'adapter.safetensors'), metadata={'format': 'pt', 'source_sha256': CHECKPOINT_SHA256, 'license': 'Apache-2.0'})
    save_file(state, str(out/'prior.safetensors'), metadata={'format': 'pt', 'source_sha256': PRIOR_SHA256, 'license': 'Apache-2.0'})
    packages = ['torch', 'transformers', 'tokenizers', 'triton', 'fla-core', 'flash-linear-attention', 'safetensors', 'huggingface-hub', 'numpy']
    modules = ['transformers.models.qwen3_5.modeling_qwen3_5', 'transformers.cache_utils',
               'transformers.integrations.sdpa_attention', 'torch.utils.checkpoint',
               'fla.modules.l2norm', 'fla.ops.utils.cache', 'triton.runtime.autotuner']
    module_hashes = {}
    for module in modules:
        parts = module.split('.'); spec = importlib.util.find_spec(parts[0])
        path = Path(next(iter(spec.submodule_search_locations))).joinpath(*parts[1:]).with_suffix('.py')
        module_hashes[module] = sha(path)
    config = json.loads((base/'config.json').read_text())
    manifest = {'schema': 1, 'model_name': 'Canopy-Jev', 'weight_license': 'Apache-2.0',
        'source_checkpoint_sha256': CHECKPOINT_SHA256, 'source_prior_sha256': PRIOR_SHA256,
        'variant': cp['variant'], 'adapter_state_sha256': state_sha(adapters), 'prior_state_sha256': state_sha(state),
        'adapter_parameters': sum(v.numel() for v in adapters.values()), 'adapter_dtype': 'float32',
        'prior_dtype': 'float32', 'prior_temperature': prior['temperature'],
        'tail': {k: v for k, v in head.items() if k != 'path'},
        'base_model': {'model_id': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'license': 'Apache-2.0', 'config_sha256': head['config_sha256'], 'index_sha256': head['index_sha256'],
            'architecture': config.get('architectures'), 'text_model_type': config['text_config'].get('model_type'),
            'tokenizer_sha256': {n: sha(base/n) for n in ['tokenizer.json', 'tokenizer_config.json']},
            'num_hidden_layers': 64, 'hidden_size': 5120,
            'weight_identity_scope': 'Exact config/index/tokenizer hashes; users supply the base snapshot. Full shard byte hashes are not included.'},
        'policy': {'exploration': .01, 'temperature': cp['temperature'], 'concentration': 256., 'alpha_floor': .001},
        'legacy_demonstrations_sha256': '2f161e50b47757e4b24b0b7d533d4c1cf7ca6486f664431d324dabcb0678a351',
        'default_template': 'zero_shot_choice', 'zero_shot_prompt_protocol': 'Exact measured JevBench input adapter',
        'legacy_template': 'legacy_five_demo requires a separately supplied SHA-bound official validation file',
        'tested_environment': {'python': sys.version.split()[0], 'cuda_toolkit': torch.version.cuda,
            'torch_build': str(torch.__version__), 'versions': {n: importlib.metadata.version(n) for n in packages},
            'library_source_sha256': module_hashes},
        'files': {name: sha(out/name) for name in ['adapter.safetensors', 'prior.safetensors']},
        'conversion': {'CPU_only': True, 'dtype_conversion': False, 'labels_exported': False,
                       'base_weights_exported': False, 'training_started': False}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    require(not torch.cuda.is_initialized(), 'Export unexpectedly initialized CUDA')
    print(json.dumps({'completed': True, 'adapter_parameters': manifest['adapter_parameters'], 'artifact_bytes': sum((out/n).stat().st_size for n in manifest['files']), 'manifest_sha256': sha(out/'manifest.json')}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True); p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    export(a.checkpoint, a.prior, a.base_model, a.out)
