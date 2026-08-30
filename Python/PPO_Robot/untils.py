import torch

def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None and network is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()