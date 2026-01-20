import torch

@torch.no_grad()
def update_ema(
    model_params: iter,
    ema_params: iter,
    rate: float,
):
    """
    Update the parameters of the EMA model using:
    ema_param = rate * ema_param + (1 - rate) * model_param
    """
    for model_param, ema_param in zip(model_params, ema_params):
        ema_param.data.mul_(rate).add_(model_param.data, alpha=1 - rate)
