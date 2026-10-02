# SPDX-License-Identifier: MIT
"""Independent pretrained decision decoder on frozen block-60 residual streams.

The original checkpoint is read-only. Only FP32 LoRA matrices in separately
loaded copies of blocks 60--63 are trainable; their base weights stay frozen.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)

START = 60


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=16, scale=32., dropout=.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = scale / rank
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        # Explicit casts also support bare BF16 inference without outer autocast.
        # Cast backward accumulates into the original FP32 master parameters.
        delta = F.linear(F.linear(self.dropout(x), self.A.to(x.dtype)), self.B.to(x.dtype))
        base = self.base(x)
        return base + delta.to(base.dtype) * self.scale


class PretrainedTail(nn.Module):
    def __init__(self, model_path, rank=16, scale=32., dropout=0., device="cuda", use_checkpoint=False):
        super().__init__()
        path = Path(model_path)
        config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        config._attn_implementation = "sdpa"
        config.dtype = torch.bfloat16
        assert config.num_hidden_layers == 64 and config.hidden_size == 5120
        self.config, self.use_checkpoint = config, use_checkpoint
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                self.layers = nn.ModuleList(Qwen3_5DecoderLayer(config, i) for i in range(START, 64))
                self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.to_empty(device=device)
        index_path = path / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        translation = {}
        prefix = "model.language_model."
        for local in self.state_dict():
            if local.startswith("layers."):
                _, number, rest = local.split(".", 2)
                original = prefix + f"layers.{START + int(number)}." + rest
            else:
                original = prefix + local
            assert original in weight_map
            translation[original] = local
        for filename in sorted({weight_map[key] for key in translation}):
            with safe_open(path / filename, framework="pt", device="cpu") as source:
                state = {local: source.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                         for key, local in translation.items() if weight_map[key] == filename}
            self.load_state_dict(state, strict=False, assign=True)
        self.requires_grad_(False)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)
        expected, _ = self.rotary_emb.compute_default_rope_parameters(config, device=device)
        torch.testing.assert_close(self.rotary_emb.inv_freq, expected, rtol=0, atol=0)
        self.base_parameter_count = sum(p.numel() for p in self.parameters())
        self.base_state_names = tuple(self.state_dict())
        # Keep DeltaNet decay/update/gating controllers untouched initially.
        selected = []
        for name, module in tuple(self.named_modules()):
            if isinstance(module, nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b", "in_proj_z")):
                selected.append(name)
                parent_name, child = name.rsplit(".", 1)
                setattr(self.get_submodule(parent_name), child, LowRankLinear(module, rank, scale, dropout))
        self.identity = {"path": str(path), "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                         "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                         "start_block": START, "end_block_exclusive": 64, "rank": rank, "scale": scale,
                         "dropout": dropout, "adapted_linears": selected,
                         "copied_frozen_parameters": self.base_parameter_count,
                         "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                         "original_backbone_trainable_parameters": 0}
        assert all(name.endswith((".A", ".B")) for name, p in self.named_parameters() if p.requires_grad)
        assert all(p.dtype == torch.float32 for p in self.parameters() if p.requires_grad)

    def adapters(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapters(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        assert set(state) == expected
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name in state:
                    p.copy_(state[name])

    def forward(self, hidden, valid):
        """Independent right-padded records; no reusable trainable cache."""
        assert hidden.dtype == torch.bfloat16 and valid.ndim == 2
        valid = valid.bool()
        assert not (valid[:, 1:] & ~valid[:, :-1]).any(), "Only contiguous right padding is supported"
        batch, length = valid.shape
        positions = torch.arange(length, device=hidden.device)[None].expand(batch, -1)
        rope = self.rotary_emb(hidden, positions[None].expand(3, -1, -1))
        causal = positions[0][None, :] <= positions[0][:, None]
        attention_mask = causal[None, None] & valid[:, None, None, :]
        h = hidden
        for layer in self.layers:
            mask = attention_mask if layer.layer_type == "full_attention" else valid
            def run(x, current=layer, current_mask=mask):
                return current(x, position_embeddings=rope, attention_mask=current_mask,
                               position_ids=positions, past_key_values=None, is_causal=False)
            h = checkpoint(run, h, use_reentrant=False) if self.training and self.use_checkpoint else run(h)
        final = self.norm(h)
        lengths = valid.sum(-1)
        assert (lengths > 0).all()
        return final[torch.arange(batch, device=h.device), lengths - 1]

    @torch.inference_mode()
    def tree(self, hidden, positions, allowed, prefix_length, spans):
        """Same causal tree semantics for the independently adapted decoder."""
        assert not self.training
        s, q = prefix_length, len(spans)
        rope = self.rotary_emb(hidden, positions[None].expand(3, -1, -1))
        h = hidden
        for layer in self.layers:
            residual = h
            x = layer.input_layernorm(h)
            if layer.layer_type == "full_attention":
                mixed, _ = layer.self_attn(x, position_embeddings=rope, attention_mask=allowed,
                                          past_key_values=None, is_causal=False)
            else:
                cache = DynamicCache(config=self.config)
                prefix = layer.linear_attn(x[:, :s], cache_params=cache, attention_mask=None)
                cached = cache.layers[layer.linear_attn.layer_idx]
                cached.conv_states = cached.conv_states.repeat_interleave(q, dim=0).clone()
                cached.recurrent_states = cached.recurrent_states.repeat_interleave(q, dim=0).clone()
                cached.max_batch_size = q
                longest = max(e-a for a, e in spans)
                branches = x.new_zeros((q, longest, x.shape[-1]))
                valid = torch.zeros((q, longest), dtype=torch.long, device=x.device)
                for b, (a, e) in enumerate(spans):
                    branches[b, :e-a] = x[0, a:e]
                    valid[b, :e-a] = 1
                branch_output = layer.linear_attn(branches, cache_params=cache, attention_mask=valid)
                mixed = torch.empty_like(x)
                mixed[:, :s] = prefix
                for b, (a, e) in enumerate(spans):
                    mixed[0, a:e] = branch_output[b, :e-a]
            h = residual + mixed
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)


def decode(final_hidden, prior, letter_ids, mask):
    state = prior["state_dict"]
    temperature = state["log_temperature"].float().exp().clamp(.2, 10) * prior["temperature"]
    with torch.autocast(device_type=final_hidden.device.type, enabled=False):
        logits = F.linear(final_hidden.float(), state["weight"], state["bias"]) / temperature
    return logits.gather(1, letter_ids).masked_fill(~mask, -1e4)
