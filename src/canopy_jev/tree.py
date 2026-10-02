# SPDX-License-Identifier: MIT
"""Original request-local numerical tree geometry; no cross-request cache."""
import torch

def pack_tree(prefix_ids, branch_ids, device):
    if not prefix_ids or not branch_ids or any(not x for x in branch_ids):
        raise ValueError("A tree needs a nonempty prefix and nonempty branches")
    s = len(prefix_ids)
    ids, groups, positions = list(prefix_ids), [0] * s, list(range(s))
    spans = []
    for q, branch in enumerate(branch_ids, 1):
        start = len(ids)
        ids.extend(branch)
        groups.extend([q] * len(branch))
        positions.extend(range(s, s + len(branch)))
        spans.append((start, len(ids)))
    groups = torch.tensor(groups, device=device)
    n = len(ids)
    causal = torch.arange(n, device=device)[None] <= torch.arange(n, device=device)[:, None]
    visible = (groups[None] == 0) | (groups[:, None] == groups[None])
    allowed = (causal & visible)[None, None]
    return torch.tensor(ids, device=device)[None], torch.tensor(positions, device=device)[None], allowed, spans
