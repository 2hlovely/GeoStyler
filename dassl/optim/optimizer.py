import torch


def build_optimizer(model, optim_cfg):
    return torch.optim.SGD(
        model.parameters(),
        lr=optim_cfg.LR,
        momentum=optim_cfg.MOMENTUM,
        weight_decay=optim_cfg.WEIGHT_DECAY
    )
