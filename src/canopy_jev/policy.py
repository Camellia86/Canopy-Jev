# SPDX-License-Identifier: MIT
"""Original FP32 letter-prior decode and exact Dirichlet-mean readout."""
def probabilities(logits, mask, exploration=0.01, temperature=1.1422139406204224):
    if exploration != 0.01 or temperature != 1.1422139406204224:
        raise ValueError('This release keeps the checkpoint policy fixed; no recalibration')
    p = logits.softmax(-1)
    p = (1-exploration)*p + exploration*mask/mask.sum(-1, keepdim=True)
    p = (256.*p+.001*mask)/(256.+.001*mask.sum(-1, keepdim=True))
    return (p.clamp_min(1e-30).log().masked_fill(~mask, -1e4)/temperature).softmax(-1)


def decode(final_hidden, prior, letter_ids, mask):
    import torch
    from torch.nn import functional as F
    state = prior['state_dict']
    temperature = state['log_temperature'].float().exp().clamp(.2, 10) * prior['temperature']
    with torch.autocast(device_type=final_hidden.device.type, enabled=False):
        logits = F.linear(final_hidden.float(), state['weight'], state['bias']) / temperature
    return logits.gather(1, letter_ids).masked_fill(~mask, -1e4)
