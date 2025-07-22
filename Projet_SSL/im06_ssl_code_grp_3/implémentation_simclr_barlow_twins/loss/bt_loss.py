import torch
import torch.nn.functional as F



# Barlow Twins loss
def barlow_twins_loss(z1, z2, batch_size, lambda_param=5e-3):
    """
    Barlow Twins loss function.

    Args:
        z (torch.Tensor): embeddings of shape (2 * batch_size, embedding_dim)
        lambda_param (float): weight for the off-diagonal term.

    Returns:
        torch.Tensor: scalar loss value.
    """
    # Normalization
    z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
    z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

    # Cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / batch_size

    # Invariance term: encourage diagonal to be 1
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

    # Redundancy reduction term: encourage off-diagonal to be 0
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()

    loss = on_diag + lambda_param * off_diag
    
    return loss