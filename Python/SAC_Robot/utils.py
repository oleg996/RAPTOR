import numpy as np
import torch


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None and network is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


def seed_everything(seed):
    """Seeds the global RNGs the env and agent actually draw from."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
