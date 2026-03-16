import torch
import torch.nn.functional as F
import numpy as np


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:

    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)


def select_confident_samples(logits: torch.Tensor, top_p: float) -> torch.Tensor:

    with torch.no_grad():
        ent = softmax_entropy(logits)  
        probs = logits.softmax(-1)
        top2 = probs.topk(2, dim=-1).values  
        gap = top2[:, 0] - top2[:, 1]  

        score = ent - gap
        num = max(1, int(logits.size(0) * float(top_p)))
        idx = torch.argsort(score, descending=False)[:num]
    return idx


def dyna_style_tta_step(model, images, style_indices, cfg):

    device = images.device
    style_indices = torch.as_tensor(style_indices, device=device, dtype=torch.long)


    orig_style = model.style_embedding[style_indices].detach().to(device)
    style_params = model.style_embedding[style_indices].to(device)
    style_params.requires_grad_(True)

    optimizer = torch.optim.Adam([style_params], lr=cfg.DYNA_STYLER.LR)
    reg_lambda = getattr(cfg.DYNA_STYLER, "LAMBDA", 0.0)

    for _ in range(cfg.DYNA_STYLER.TTA_STEPS):
        optimizer.zero_grad()

        logits = model.forward_tta(images, style_params, style_indices)
        idx = select_confident_samples(logits, cfg.DYNA_STYLER.SELECTION_P)
        if idx.numel() == 0:
            continue
        entropy_loss = softmax_entropy(logits[idx]).mean()
        anchor_loss = F.mse_loss(style_params, orig_style)
        loss = entropy_loss + reg_lambda * anchor_loss
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        model.style_embedding[style_indices] = style_params.detach()

    return model
