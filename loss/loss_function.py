import torch.nn as nn

def anchor_loss(device, input, target, loss_type='softmax'):
    if loss_type == "l2":
        loss_func = nn.MSELoss(reduction="mean")
        loss = loss_func(input, target)
    elif loss_type == "softmax":
        B, C, H, W = input.shape
        criterion = nn.BCELoss(reduction='none').to(device)
        loss = criterion(nn.functional.softmax(input, dim=1), target)
        loss = loss.sum() / (B*H*W + 1e-10)
    return loss