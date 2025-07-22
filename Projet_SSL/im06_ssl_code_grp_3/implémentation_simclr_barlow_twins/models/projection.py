import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection head avec un MLP comme décrit dans le papier SimCLR.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.BatchNorm1d(output_dim)
        # )

        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.BatchNorm1d(int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.BatchNorm1d(int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)
    


class ProjectionHead2(nn.Module):
    """
    Projection head avec un MLP comme décrit dans le papier Barlow Twins.
    """
    def __init__(self, input_dim=2048, proj_dim=8192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)