# SPDX-License-Identifier: MIT
"""Portable lean first60 + independently adapted last4 inference loader."""
import json
import os
from pathlib import Path
import threading
import time

from .artifacts import load_manifest, require, resolve_base_model, sha, state_sha, verify_base_model, verify_runtime_sources
from .prompt import prepare


def validate_geometry(prefix, branches, counts, letters, limit):
    require(isinstance(prefix, list) and bool(prefix) and isinstance(branches, list) and bool(branches)
            and all(isinstance(b, list) and bool(b) for b in branches), 'Nonempty prefix and branches required')
    require(all(type(t) is int and t >= 0 for s in [prefix, *branches] for t in s), 'Invalid token IDs')
    require(len(counts) == len(branches) and all(type(k) is int and 1 <= k <= 10 for k in counts), 'Invalid candidate counts')
    require(len(prefix)+sum(map(len, branches)) <= limit, 'Request exceeds packed-token limit; no truncation')
    letters = [list(range(k)) for k in counts] if letters is None else letters
    require(len(letters) == len(counts) and all(len(v) == k and len(set(v)) == k
            and all(type(i) is int and 0 <= i < 10 for i in v) for v, k in zip(letters, counts)), 'Invalid native letter axis')
    return letters


class CanopyJev:
    """Numerical Choice model. Noul and Score use explicit candidate adapters.

    Loading requires an exact compatible base snapshot and at least 64 GiB of
    free NVIDIA GPU memory. No model is loaded by importing this package.
    """
    @classmethod
    def from_pretrained(cls, artifact_dir, *, base_model, revision=None, device='cuda:0',
                        local_files_only=False, min_free_gib=64., max_tokens=16384):
        return cls(artifact_dir, base_model=base_model, revision=revision, device=device,
                   local_files_only=local_files_only, min_free_gib=min_free_gib, max_tokens=max_tokens)

    def __init__(self, artifact_dir, *, base_model, revision=None, device='cuda:0',
                 local_files_only=False, min_free_gib=64., max_tokens=16384):
        started = time.perf_counter()
        require(min_free_gib >= 64., 'The supported loading guard is at least 64 GiB free')
        require(type(max_tokens) is int and 1 <= max_tokens <= 16384, 'max_tokens must be 1..16384')
        manifest = load_manifest(artifact_dir)
        original = resolve_base_model(base_model, revision, local_files_only)
        verify_base_model(original, manifest)
        verify_runtime_sources(manifest)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel
        from .tail import PretrainedTail
        from .stem import FrozenTreeStem
        from .runtime_pin import pin_l2norm_runtime, runtime_contract
        self.device = torch.device(device)
        require(self.device.type == 'cuda' and torch.cuda.is_available(), 'CUDA inference is required; no CPU model backend is supplied')
        self.max_tokens = max_tokens
        self._lock = threading.RLock()
        with torch.cuda.device(self.device):
            require(torch.cuda.mem_get_info(self.device)[0] >= min_free_gib*1024**3,
                    'Insufficient free GPU memory; at least 64 GiB is required before loading')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            pin_l2norm_runtime()
            adapters = load_file(str(Path(artifact_dir)/'adapter.safetensors'), device='cpu')
            prior_state = load_file(str(Path(artifact_dir)/'prior.safetensors'), device='cpu')
            require(state_sha(adapters) == manifest['adapter_state_sha256'], 'Adapter tensor identity mismatch')
            require(state_sha(prior_state) == manifest['prior_state_sha256'], 'Prior tensor identity mismatch')
            require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in adapters.values()), 'Adapters must retain original FP32 masters')
            require(set(prior_state) == {'weight', 'bias', 'log_temperature'} and prior_state['weight'].shape == (10, 5120)
                    and prior_state['bias'].shape == (10,) and prior_state['log_temperature'].shape == (), 'Wrong prior tensor shapes')
            require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in prior_state.values()), 'Invalid FP32 prior')
            self.prior = {'state_dict': {k: v.to(self.device) for k, v in prior_state.items()},
                          'temperature': manifest['prior_temperature']}
            config = AutoConfig.from_pretrained(original, local_files_only=True).text_config
            config._attn_implementation = 'sdpa'; config.dtype = torch.bfloat16
            previous = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device('meta'):
                    view = Qwen3_5TextModel(config)
            finally:
                torch.set_default_dtype(previous)
            hid = manifest['tail']
            tail = PretrainedTail(model_path=str(original), rank=hid['rank'], scale=hid['scale'],
                                 dropout=hid['dropout'], device=str(self.device), use_checkpoint=False)
            for key, value in hid.items():
                require(tail.identity[key] == value, 'Tail structure mismatch: '+key)
            tail.load_adapters(adapters)
            self.adapter_names = {n for n, p in tail.named_parameters() if p.requires_grad}
            require(state_sha(tail.adapters()) == manifest['adapter_state_sha256'], 'Loaded adapters changed dtype/value')
            tail.eval().requires_grad_(False)
            view.layers = torch.nn.ModuleList([*list(view.layers[:60]), *tail.layers])
            view.norm = tail.norm; view.rotary_emb = tail.rotary_emb
            needed = {n for n in view.state_dict() if n == 'embed_tokens.weight'
                      or (n.startswith('layers.') and int(n.split('.')[1]) < 60)}
            index = json.loads((original/'model.safetensors.index.json').read_text())['weight_map']
            prefix = 'model.language_model.'
            require(all(prefix+n in index for n in needed), 'Missing first60 base tensors')
            loaded = set()
            for filename in sorted({index[prefix+n] for n in needed}):
                with safe_open(original/filename, framework='pt', device='cpu') as source:
                    values = {n: source.get_tensor(prefix+n).to(device=self.device, dtype=torch.bfloat16)
                              for n in needed if index[prefix+n] == filename}
                    view.load_state_dict(values, strict=False, assign=True); loaded.update(values)
                del values
            require(loaded == needed and not any(t.is_meta for t in [*view.parameters(), *view.buffers()]), 'Incomplete model materialization')
            view.eval().requires_grad_(False)
            self.view, self.tail = view, tail
            self.stem = FrozenTreeStem(view, max_tokens=max_tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True)
            self.tokenizer.padding_side = 'right'
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.manifest = manifest
            self._versions = {n: (p.data_ptr(), p._version) for n, p in view.named_parameters()}
            self.identity = {'artifact_manifest_sha256': sha(Path(artifact_dir)/'manifest.json'),
                'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
                'source_prior_sha256': manifest['source_prior_sha256'],
                'adapter_state_sha256': manifest['adapter_state_sha256'],
                'base_model': manifest['base_model'], 'runtime': runtime_contract(),
                'physical_parameter_bytes': sum(p.numel()*p.element_size() for p in view.parameters()),
                'parameters_frozen': True, 'cached_h60_used': False, 'vocabulary_head_loaded': False,
                'numerical_domain': 'fresh request-local shared-prefix tree, not canonical B1 or historical cached features',
                'probability_policy': manifest['policy'], 'default_template': 'zero_shot_choice',
                'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda}
            torch.cuda.synchronize(self.device)
        self.identity['load_seconds'] = time.perf_counter()-started

    def assert_unchanged(self):
        require(all(not p.requires_grad for p in self.view.parameters()), 'Model parameters became trainable')
        require(self._versions == {n: (p.data_ptr(), p._version) for n, p in self.view.named_parameters()}, 'Base or adapter parameter versions changed')
        require(state_sha({n: self.tail.get_parameter(n) for n in self.adapter_names}) == self.manifest['adapter_state_sha256'], 'Adapter values changed')

    def infer_tokens(self, prefix, branches, candidate_counts, letter_ids=None):
        """Return original CPU float32 [Q,10] probabilities from fresh full64 work."""
        letters = validate_geometry(prefix, branches, candidate_counts, letter_ids, self.max_tokens)
        import torch
        from .policy import decode, probabilities
        from .runtime_pin import assert_runtime_pin
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            assert_runtime_pin()
            h60, positions, allowed, spans = self.stem(prefix, branches)
            hidden = self.tail.tree(h60, positions, allowed, len(prefix), spans)
            final = torch.stack([hidden[0, end-1] for _, end in spans])
            ids = torch.zeros(len(branches), 10, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, values in enumerate(letters):
                ids[i, :len(values)] = torch.tensor(values, device=self.device); mask[i, :len(values)] = True
            p = probabilities(decode(final, self.prior, ids, mask), mask,
                              self.manifest['policy']['exploration'], self.manifest['policy']['temperature'])
            require(p.dtype == torch.float32 and p.shape == (len(branches), 10) and torch.isfinite(p).all(), 'Invalid output tensor')
            require(torch.allclose(p.sum(-1), torch.ones(len(branches), device=self.device), atol=1e-6)
                    and (p[~mask] == 0).all(), 'Probability normalization/mask failure')
            return p.cpu().numpy()

    def predict(self, state, questions, *, template='zero_shot_choice', demonstrations_path=None):
        """Typed JSON-ready decisions; arbitrary target/reference fields are ignored.

        Noul is a two-candidate adaptation; Score is an ordered-candidate adaptation.
        Their probabilities come from the same Choice head, not separate heads.
        """
        demos = None
        if demonstrations_path is not None:
            require(template == 'legacy_five_demo', 'Demonstrations are allowed only in explicit legacy mode')
            require(sha(demonstrations_path) == self.manifest['legacy_demonstrations_sha256'], 'Legacy demonstration file checksum mismatch')
            demos = [json.loads(s) for s in Path(demonstrations_path).read_text().splitlines() if s.strip()]
        rows, prefix, branches = prepare(self.tokenizer, state, questions, template, demos)
        p = self.infer_tokens(prefix, branches, [len(r['options']) for r in rows])
        answers = []
        for i, row in enumerate(rows):
            k = len(row['keys']); values = p[i, :k].tolist(); best = int(p[i, :k].argmax())
            result = {'id': row['id'], 'type': row['type'], 'choice_index': best, 'choice_key': row['keys'][best],
                      'probabilities': dict(zip(row['keys'], values)), 'generated_tokens': 0}
            if row['type'] == 'noul':
                result['value'] = row['keys'][best] == 'yes'; result['probability_true'] = values[1]
            elif row['type'] == 'score':
                result['value'] = best; result['expected_level'] = sum(j*x for j, x in enumerate(values))
            answers.append(result)
        return {'model': 'Canopy-Jev', 'template': template, 'questions': answers,
                'computation': {'shared_prefix_tokens': len(prefix), 'branch_tokens': list(map(len, branches)),
                                'packed_tokens': len(prefix)+sum(map(len, branches)), 'generated_tokens': 0,
                                'numerical_domain': self.identity['numerical_domain']},
                'candidate_adaptation': 'Noul and Score use the same at-most-10-candidate Choice head',
                'tie_break': 'First candidate at the maximum probability'}

    __call__ = predict
