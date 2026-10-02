# SPDX-License-Identifier: MIT
"""Small safe-tensor artifact and exact base-model identity validation."""
import hashlib
import json
from pathlib import Path

CHECKPOINT_SHA256 = '506410609a6ba6eb422182fddda095b81fa4923011ceaa3717e85a74817237f4'
PRIOR_SHA256 = '4620b120f1887a2240dc0cc748de9631b701e8db16a200450d3b41b4e25d6bee'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_manifest(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    require(manifest['schema'] == 1, 'Unsupported artifact schema')
    require(manifest['source_checkpoint_sha256'] == CHECKPOINT_SHA256, 'Wrong adapter checkpoint')
    require(manifest['source_prior_sha256'] == PRIOR_SHA256, 'Wrong decision prior')
    require(set(manifest['files']) == {'adapter.safetensors', 'prior.safetensors'}, 'Unexpected artifact file set')
    for name, digest in manifest['files'].items():
        require(sha(folder / name) == digest, 'Artifact checksum mismatch: ' + name)
    require(manifest['policy'] == {'exploration': 0.01, 'temperature': 1.1422139406204224,
                                   'concentration': 256.0, 'alpha_floor': 0.001}, 'Unsupported probability policy')
    return manifest


def resolve_base_model(value, revision=None, local_files_only=False):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=value, revision=revision, local_files_only=local_files_only))


def verify_base_model(path, manifest):
    path = Path(path)
    expected = manifest['base_model']
    for filename, key in [('config.json', 'config_sha256'), ('model.safetensors.index.json', 'index_sha256')]:
        require(sha(path / filename) == expected[key], 'Base model identity mismatch: ' + filename)
    for filename, digest in expected['tokenizer_sha256'].items():
        require(sha(path / filename) == digest, 'Tokenizer identity mismatch: ' + filename)
    config = json.loads((path / 'config.json').read_text())['text_config']
    require(config['num_hidden_layers'] == 64 and config['hidden_size'] == 5120,
            'Expected the exact 64-layer, width-5120 text backbone')
    return config


def state_sha(state):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_runtime_sources(manifest):
    """Fail explicitly on an unreviewed numerical-library build."""
    import importlib.util
    for name, expected in manifest['tested_environment']['library_source_sha256'].items():
        parts = name.split('.')
        spec = importlib.util.find_spec(parts[0])
        require(spec is not None and spec.submodule_search_locations is not None, 'Missing runtime package: '+parts[0])
        path = Path(next(iter(spec.submodule_search_locations))).joinpath(*parts[1:]).with_suffix('.py')
        require(path.is_file() and sha(path) == expected, 'Untested numerical library source: '+name+'. Use the tested build; version numbers alone may be insufficient.')
