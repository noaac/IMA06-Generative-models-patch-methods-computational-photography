import os
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """Fixe toutes les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Pour multi-GPU

    torch.backends.cudnn.deterministic = True  # Déterminisme
    torch.backends.cudnn.benchmark = False     # Pas d'optimisation heuristique

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Seed fixée à {seed}")


def print_config(args):
    """Affiche la configuration complète de l'entraînement SimCLR"""
    print("=" * 60)
    print("CONFIGURATION SIMCLR")
    print("=" * 60)

    # Device
    print(f"Device: {args.device}")

    # Dataset
    print(f"Dataset: {args.dataset} (Image size: {args.image_size}x{args.image_size})")
    print(f"Number of classes: {args.num_classes}")
    print(f"Download dataset: {args.download}")

    # Experiment setup
    print(f"Training enabled: {args.train}")
    print(f"Evaluation enabled: {args.evaluate}")
    print(f"Resume from checkpoint: {args.checkpoint}")

    # Encoder
    print(f"Encoder: {args.encoder}")
    print(f"Pretrained encoder: {args.pretrained}")
    print(f"Sync BatchNorm: {args.sync_batch_norm}")

    # Projection Head
    print("Projection Head:")
    print(f"  Input dim: {args.proj_input_dim}")
    print(f"  Hidden dim: {args.proj_hidden_dim}")
    print(f"  Output dim: {args.proj_output_dim}")

    # Training
    print("Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Pin memory: {args.pin_memory}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Vectorized loss: {args.vectorized_loss}")
    print(f"  GPU transforms: {args.gpu_transforms}")
    print(f"  Use LARS: {args.use_lars}")

    # Warmup & Scheduling
    print("Warmup & LR Scheduling:")
    print(f"  Warmup enabled: {args.warmup}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Cosine LR decay: {args.cosine_decay}")

    # Evaluation
    print("Evaluation Configuration:")
    print(f"  Eval epochs: {args.eval_epochs}")
    print(f"  Eval batch size: {args.eval_batch_size}")
    print(f"  Eval learning rate: {args.eval_lr}")
    print(f"  Eval weight decay: {args.eval_weight_decay}")
    print(f"  Cut ratio: {args.cut_ratio}")
    print(f"  Evaluate from checkpoint: {args.eval_from_ckpt}")

    # Logging & Checkpointing
    print("Logging & Checkpointing:")
    print(f"  Log interval (batches): {args.log_interval}")
    print(f"  Save checkpoint every: {args.save_every} epochs")

    print("=" * 60)

################## WANDB ################################################### WANDB #########################################
############################################################################################################################
def get_wandb_config(args):
    """Configuration pour WandB"""
    return {
        # Modèle
        "encoder_name": args.encoder,
        "projection_input_dim": args.proj_input_dim,
        "projection_hidden_dim": args.proj_hidden_dim,
        "projection_output_dim": args.proj_output_dim,
        "pretrained": args.pretrained,
        "sync_batch_norm": args.sync_batch_norm,

        # Entraînement
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "use_lars": args.use_lars,
        "warmup": args.warmup,
        "warmup_epochs": args.warmup_epochs,
        "cosine_decay": args.cosine_decay,

        # Dataset
        "dataset_name": args.dataset,
        "image_size": args.image_size,
        "num_classes": args.num_classes,

        # Optimisations
        "use_vectorized_loss": args.vectorized_loss,
        "use_gpu_transforms": args.gpu_transforms,

        # Évaluation
        "evaluation_epochs": args.eval_epochs,
        "evaluation_batch_size": args.eval_batch_size,
        "evaluation_lr": args.eval_lr,
        "cut_ratio": args.cut_ratio,
        "evaluation_from_checkpoint": args.eval_from_ckpt,

        # Logging
        "wandb_dir": args.wandb_dir,
        "wandb_run_name": args.wandb_run_name,
        "wandb_tags": args.wandb_tags,

        "log_performance_metrics": True,
        "log_time_breakdown": True,
        "performance_dashboard_frequency": 5,
    }
############################################################################################################################
################## WANDB ################################################### WANDB ######################################### 