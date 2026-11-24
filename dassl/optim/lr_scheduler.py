import torch


def build_lr_scheduler(optimizer, optim_cfg, max_epoch=None):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0
    )
