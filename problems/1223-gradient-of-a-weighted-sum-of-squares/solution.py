import torch

def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    # TODO
    w = torch.tensor(w_list,requires_grad=True,dtype=float)
    x = torch.tensor(x_list,dtype=float)
    y = torch.sum((w*x)**2)
    L = 0.5*y
    L.backward()
    return [round(x,4) for x in w.grad.tolist()]
    pass
