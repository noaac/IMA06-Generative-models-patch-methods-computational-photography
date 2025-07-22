import os
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import get_wandb_config
from sklearn.metrics.pairwise import cosine_similarity


def init_wandb(args, config=None, resume_run_id=None):
    """Initialise WandB"""
    if not args.use_wandb:
        return None
    if config is None:
        config = get_wandb_config(args)
    run_name = args.wandb_run_name
    if run_name is None:
        slurm_id = os.environ.get('SLURM_JOB_ID', 'local')
        run_name = f"simclr_bs{args.batch_size}_lr{args.lr:.3f}_temp{args.temperature}_imagesize{args.image_size}"
        
    # partie pour adaptation sur slurm
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = args.wandb_dir
    os.environ['WANDB_SILENT'] = 'true'
    
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        resume="allow" if resume_run_id else None,
        id=resume_run_id)
    
    print(f"Initialisation de Wandb en mode {args.wandb_mode}. Données enregistrées dans {args.wandb_dir}.\n")
    return run

def log_metrics(metrics_dict, args, step=None):
    """Log des métriques sur WandB"""
    if args.use_wandb and wandb.run is not None:
        wandb.log(metrics_dict, step=step)

def log_model_artifact(model_path, args, model_name="simclr_model"):
    """Sauvegarde le modèle comme artifact WandB"""
    if args.use_wandb and wandb.run is not None and args.wandb_save_model:
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
def log_performance_dashboard(history, args, phase="training"):
    """Crée un dashboard de performance détaillé"""
    if not args.use_wandb or wandb.run is None:
        return
    avg_transform = sum(history['transform_times']) / len(history['transform_times'])
    avg_forward = sum(history['forward_times']) / len(history['forward_times'])
    avg_backward = sum(history['backward_times']) / len(history['backward_times'])
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    times = [avg_transform, avg_forward, avg_backward]
    labels = ['Transforms', 'Forward', 'Backward']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'{phase.capitalize()} Time Distribution')
    
    plt.subplot(1, 3, 2)
    batches = range(len(history['batch_times']))
    plt.plot(batches, history['transform_times'], label='Transform', alpha=0.7)
    plt.plot(batches, history['forward_times'], label='Forward', alpha=0.7)
    plt.plot(batches, history['backward_times'], label='Backward', alpha=0.7)
    plt.xlabel('Batch')
    plt.ylabel('Time (s)')
    plt.title('Time Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist([history['transform_times'], history['forward_times'], history['backward_times']], 
             bins=20, alpha=0.7, label=['Transform', 'Forward', 'Backward'])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.title('Time Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({f"{phase}_performance_dashboard": wandb.Image(plt)})
    plt.close()

def log_learning_curves(history, args, phase="training"):
    """Crée et log des courbes d'apprentissage"""
    if not args.use_wandb or wandb.run is None:
        return
    
    if 'losses' in history:
        # Courbe de loss
        plt.figure(figsize=(10, 6))
        plt.plot(history['losses'])
        plt.title(f'{phase.capitalize()} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        wandb.log({f"{phase}_loss_curve": wandb.Image(plt)})
        plt.close()
    
    if 'accuracies' in history:
        # Courbe d'accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history['accuracies'])
        plt.title(f'{phase.capitalize()} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        wandb.log({f"{phase}_accuracy_curve": wandb.Image(plt)})
        plt.close()

def log_confusion_matrix(y_true, y_pred, args, class_names=None):
    """Log matrice de confusion"""
    if args.use_wandb and wandb.run is not None:

        y_true = [yt[0] if isinstance(yt, list) else yt for yt in y_true]
        y_pred = [yp[0] if isinstance(yp, list) else yp for yp in y_pred]

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names)
        })

def log_training_images(images, args, predictions=None, max_images=8):
    """Log des images d'entraînement avec prédictions"""
    if not args.use_wandb or wandb.run is None:
        return
    
    # Limite le nombre d'images
    images = images[:max_images]
    
    wandb_images = []
    for i, img in enumerate(images):
        # Convertit tensor vers numpy si nécessaire
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        
        caption = f"Sample {i}"
        if predictions is not None:
            caption += f" | Pred: {predictions[i]}"
        
        wandb_images.append(wandb.Image(img, caption=caption))
    
    wandb.log({"training_samples": wandb_images})

def finish_wandb(args):
    """Termine le run WandB et prépare pour sync"""
    if args.use_wandb and wandb.run is not None:
        run_id = wandb.run.id
        wandb.finish()
        print(f"\nWandB run terminé (ID: {run_id}). Données offline dans: {args.wandb_dir}")
        if 'SLURM_JOB_ID' in os.environ:
            print("Les instructions de sync seront affichées à la fin du job")
        else:
            print("Pour sync: wandb sync wandb_offline/wandb/offline-run-*")

# fonction pour analyser la qualité des représentations : ultra important !
def log_representation_quality(model, data_loader, args, phase="training"):
    if not args.use_wandb or not hasattr(args, 'wandb_run') or args.wandb_run is None:
        return
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= 10:  # Limite pour performance
                break
            x = x.to(args.device)
            h, z = model(x)
            embeddings.append(z.cpu())
            labels.append(y)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Distribution des normes d'embeddings
    norms = torch.norm(embeddings, dim=1)
    plt.figure(figsize=(8, 6))
    plt.hist(norms.numpy(), bins=50, alpha=0.7)
    plt.title(f'{phase} - Distribution des normes d\'embeddings')
    plt.xlabel('Norme L2')
    plt.ylabel('Fréquence')
    
    wandb.log({f"{phase}/embedding_norms_distribution": wandb.Image(plt)})
    plt.close()
    
    # Similarité intra-classe vs inter-classe
    similarities = cosine_similarity(embeddings.numpy())
    intra_class_sims = []
    inter_class_sims = []
    for i in range(len(labels)):
        for j in range(i+1, min(i+100, len(labels))):  # Limite pour performance
            if labels[i] == labels[j]:
                intra_class_sims.append(similarities[i, j])
            else:
                inter_class_sims.append(similarities[i, j])
    
    if intra_class_sims and inter_class_sims:
        wandb.log({
            f"{phase}/avg_intra_class_similarity": np.mean(intra_class_sims),
            f"{phase}/avg_inter_class_similarity": np.mean(inter_class_sims),
            f"{phase}/similarity_separation": np.mean(intra_class_sims) - np.mean(inter_class_sims)
        })
    
    model.train()  # Remettre en mode training

def log_hyperparameter_impact(args):
    # Log l'impact des hyperparamètres principaux
    if not args.use_wandb:
        return
    config_summary = {
        "config/batch_size": args.batch_size,
        "config/temperature": args.temperature,
        "config/learning_rate": getattr(args, 'lr', 'auto'),
        "config/warmup_epochs": getattr(args, 'warmup_epochs', 0),
        "config/total_epochs": args.epochs,
        "config/encoder": getattr(args, 'encoder', 'resnet50'),
        "config/projection_dim": getattr(args, 'proj_output_dim', 128),
        "config/use_lars": getattr(args, 'use_lars', False),
        "config/weight_decay": getattr(args, 'weight_decay', 1e-6),
        "config/vectorized_loss": getattr(args, 'vectorized_loss', True),
        "config/gpu_transforms": getattr(args, 'gpu_transforms', True),
        "config/transform_crop": args.transform_crop,
        "config/transform_color": args.transform_color,
        "config/transform_blur": args.transform_blur,
        "config/transform_rotation": args.transform_rotation,
    }
    wandb.log(config_summary)
