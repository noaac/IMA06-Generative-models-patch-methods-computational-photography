import torch
import torch.nn as nn
from torchvision import models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet import create_encoder
from models.projection import ProjectionHead

class SimCLR(nn.Module):
    """
    Modèle SimCLR principal qui combine :
    - L'encoder : ici on a pris un Resnet50 dont on a retiré la dernière couche.
    - Projection Head : ici un MLP avec une couche cachée.
    """
    
    def __init__(self, encoder, projection_head, projection_input_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head


        self.linear_classifier = nn.Linear(projection_input_dim, num_classes)
        self.evaluation = False
        
    def forward(self, x):
        """
        Args : 
            x : Tensor avec les images pour une passe de l'entraînement. 
            De taille (batch_size, 3, taille_image, taille_image)
        Returns :
            h: sortie de l'encoder (batch_size, encoder_dim)
            z: sortie de la projection head (batch_size, projection_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        if not self.evaluation:
            z = self.projection_head(h)
            return h, z
        else:
            z = self.linear_classifier(h)
            return z

    
def create_simclr_model(args):
    
    encoder = create_encoder(pretrained=args.pretrained)
    projection_head = ProjectionHead(input_dim=args.proj_input_dim, hidden_dim=args.proj_hidden_dim, output_dim=args.proj_output_dim)
    model = SimCLR(encoder, projection_head, args.proj_input_dim, args.num_classes)

    if args.sync_batch_norm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    return model