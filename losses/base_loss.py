import torch


def compute_mse(pred, gt, mask=None):
    """
    Compute MSE between pred and gt with optional mask.
    Mask can be any shape broadcastable to pred.
    """
    if mask is None:
        return torch.mean((pred - gt) ** 2)

    # Broadcast mask
    mask = mask.to(pred.dtype)
    mask_b = torch.broadcast_to(mask, pred.shape)

    num = torch.sum((pred - gt) ** 2 * mask_b)
    den = torch.sum(mask_b) + 1e-8
    return num / den


def compute_mae(pred, gt, mask=None):
    """
    Compute MAE between pred and gt with optional mask.
    Mask can be any shape broadcastable to pred.
    """
    if mask is None:
        return torch.mean(torch.abs(pred - gt))

    # Broadcast mask
    mask = mask.to(pred.dtype)
    mask_b = torch.broadcast_to(mask, pred.shape)

    num = torch.sum(torch.abs(pred - gt) * mask_b)
    den = torch.sum(mask_b) + 1e-8
    return num / den


def compute_depth_loss(pred_depth, gt_depth):
    """
    Compute the shift- and scale-invariant depth loss from
    https://arxiv.org/pdf/1907.01341.pdf.
    """
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt

    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))
