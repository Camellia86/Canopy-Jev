# SPDX-License-Identifier: MIT
"""Frozen, isolated shared-prefix computation up to the input of block 60."""
import torch
from transformers import DynamicCache

from .tree import pack_tree
from .tail import START


class FrozenTreeStem:
    def __init__(self, model, max_tokens=16384):
        self.model, self.max_tokens = model, max_tokens
        assert len(model.layers) == 64 and not any(p.requires_grad for p in model.parameters())

    @torch.inference_mode()
    def __call__(self, prefix_ids, branch_ids, include_output=False):
        model = self.model
        device = next(model.parameters()).device
        ids, positions, allowed, spans = pack_tree(prefix_ids, branch_ids, device)
        assert ids.shape[1] <= self.max_tokens
        s, q = len(prefix_ids), len(branch_ids)
        h = model.embed_tokens(ids)
        rope = model.rotary_emb(h, positions[None].expand(3, -1, -1))
        saved = None
        stop = 64 if include_output else START
        for i, layer in enumerate(model.layers[:stop]):
            if i == START:
                saved = h
            residual = h
            x = layer.input_layernorm(h)
            if model.config.layer_types[i] == "full_attention":
                mixed, _ = layer.self_attn(x, position_embeddings=rope, attention_mask=allowed,
                                          past_key_values=None, is_causal=False)
            else:
                cache = DynamicCache(config=model.config)
                prefix = layer.linear_attn(x[:, :s], cache_params=cache, attention_mask=None)
                state = cache.layers[i]
                state.conv_states = state.conv_states.repeat_interleave(q, dim=0).clone()
                state.recurrent_states = state.recurrent_states.repeat_interleave(q, dim=0).clone()
                state.max_batch_size = q
                longest = max(e-a for a, e in spans)
                branches = x.new_zeros((q, longest, x.shape[-1]))
                valid = torch.zeros((q, longest), dtype=torch.long, device=device)
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
        # This is an unnormalized residual stream, exactly block index 60 input.
        if include_output:
            assert saved is not None
            return saved, positions, allowed, spans, model.norm(h)
        return h, positions, allowed, spans
