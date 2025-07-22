import torch
import torch.nn.functional as F


def sim(u,v):
    return torch.sum(u*v,dim=-1)/((torch.norm(u, dim=-1)*torch.norm(v, dim=-1)))


def contrastive_loss_1(z, batch_size, temperature=0.1):
    total_loss = 0.0
    for i in range(2 * batch_size):
        if i < batch_size:
            z_i = z[i]
            z_j = z[i + batch_size]
        else:
            z_i = z[i - batch_size]
            z_j = z[i]
        sim_ij = sim(z_i,z_j)
        sim_all = torch.tensor([sim(z_i, z[k]) for k in range(2 * batch_size) if k != i])
        
        loss = -torch.log(
            torch.exp(sim_ij / temperature) / 
            torch.sum(torch.exp(sim_all/temperature))
        )
        total_loss += loss
    return total_loss / (2 * batch_size)


def contrastive_loss_vectorized(z, batch_size, temperature=0.1):
    device = z.device # 
    z_norm = F.normalize(z, dim=1)
    sim_matrix = torch.mm(z_norm, z_norm.t()) / temperature
    labels = torch.cat([
        torch.arange(batch_size, 2*batch_size, device=device),
        torch.arange(0, batch_size, device=device)
    ])
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, -float('inf'))
    return F.cross_entropy(sim_matrix, labels)

def contrastive_loss(z, batch_size, temperature=0.1, use_vectorized=True):
    """
    Fonction qu'on appelle lors de l'entraînement pour calculer la loss contrastive.
    On peut choisir entre la version initiale et la version boostée
    """
    if use_vectorized: return contrastive_loss_vectorized(z,batch_size,temperature)
    else: return contrastive_loss_1(z, batch_size, temperature)