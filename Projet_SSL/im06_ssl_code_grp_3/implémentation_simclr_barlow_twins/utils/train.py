import time
import torch
from tqdm import tqdm
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.contrastive import contrastive_loss
from data.transforms import apply_transforms_to_batch
from torch.optim.lr_scheduler import LambdaLR
from utils.multi_gpu_utils import is_main_process
from utils.wandb_utils import log_representation_quality


from utils.wandb_utils import log_metrics, log_model_artifact, log_learning_curves, log_performance_dashboard

def train_simclr(model, train_loader, optimizer, transform_pipeline, num_epochs, args, checkpoints_dir, logs_dir, train_sampler):
    """
    Fonction d'entraînement de notre SimCLR
    On y a ajouté : 
        - des time pour observer les performances
        - un dictionnaire history qui stocke différentes métriques
    """
    if is_main_process():
        print(f"Device: {args.device}")
    model.to(args.device)
    if args.gpu_transforms and hasattr(transform_pipeline, 'to'):
        transform_pipeline = transform_pipeline.to(args.device)
    model.train()

    if args.cosine_decay or args.warmup:
        total_steps = num_epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)

    if not args.warmup and args.cosine_decay:
        def lr_lambda(step):
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda)

    elif args.warmup and not args.cosine_decay:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return optimizer.param_groups[0]['lr']
        scheduler = LambdaLR(optimizer, lr_lambda)

    elif args.warmup and args.cosine_decay:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda)

    else:
        scheduler = None


    history = {
        'losses': [],
        'epoch_times': [],
        'batch_times': [],
        'transform_times': [],
        'forward_times': [],
        'backward_times': []
    }
    total_start_time = time.time()

    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        
        if is_main_process():
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 50)
        
        for batch_idx, (x, labels) in enumerate(tqdm(train_loader, desc="Training")):
            batch_start_time = time.time()
            
            x = x.to(args.device)

            transform_start = time.time()
            x_i = apply_transforms_to_batch(x, transform_pipeline, use_gpu=args.gpu_transforms)
            x_j = apply_transforms_to_batch(x, transform_pipeline, use_gpu=args.gpu_transforms)
            
            if not args.gpu_transforms:
                x_i = x_i.to(args.device)
                x_j = x_j.to(args.device)
            
            transform_time = time.time() - transform_start
            history['transform_times'].append(transform_time)
            
            forward_start = time.time()
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)
            
            z = torch.cat((z_i, z_j), dim=0)
            
            loss = contrastive_loss(z, x_i.size(0), args.temperature, use_vectorized=args.vectorized_loss)
            forward_time = time.time() - forward_start
            history['forward_times'].append(forward_time)

            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            backward_time = time.time() - backward_start
            history['backward_times'].append(backward_time)
            
            total_loss += loss.item()
            batch_time = time.time() - batch_start_time
            history['batch_times'].append(batch_time)

            if args.warmup and global_step < warmup_steps:
                in_warmup = True
            else:
                in_warmup = False

            global_step += 1
            
            if is_main_process() and batch_idx % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Warmup: {in_warmup} | "
                      f"LR: {current_lr:.6f} | "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Transform: {transform_time:.3f}s | "
                      f"Forward: {forward_time:.3f}s | "
                      f"Backward: {backward_time:.3f}s | "
                      f"Total: {batch_time:.3f}s")

            if is_main_process() and batch_idx % args.log_interval_wandb == 0:
                contrastive_metrics = compute_detailed_contrastive_metrics(z_i, z_j, args.temperature, args)
                metrics = {
                    'batch_loss': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'batch': batch_idx,
                    'temperature': args.temperature,
                    
                    'perf/transform (s)': transform_time,
                    'perf/forward (s)': forward_time,
                    'perf/backward (s)': backward_time,
                    'perf/batch (s)': batch_time,
                
                    'perf/transform (%)': (transform_time / batch_time) * 100,
                    'perf/forward (%)': (forward_time / batch_time) * 100,
                    'perf/backward (%)': (backward_time / batch_time) * 100,
                    
                    **contrastive_metrics
                }
                
                log_metrics(metrics, args, step=global_step)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)

        epoch_avg_transform = sum(history['transform_times'][-len(train_loader):]) / len(train_loader)
        epoch_avg_forward = sum(history['forward_times'][-len(train_loader):]) / len(train_loader)
        epoch_avg_backward = sum(history['backward_times'][-len(train_loader):]) / len(train_loader)
        epoch_avg_batch = sum(history['batch_times'][-len(train_loader):]) / len(train_loader)
        
        epoch_metrics = {
            'epoch_loss': avg_loss,
            'epoch': epoch,
            'epoch_time': epoch_time,
            
            'perf_epoch/avg_transform_time': epoch_avg_transform,
            'perf_epoch/avg_forward_time': epoch_avg_forward,
            'perf_epoch/avg_backward_time': epoch_avg_backward,
            'perf_epoch/avg_batch_time': epoch_avg_batch,
        }
        
        if is_main_process():
            log_metrics(epoch_metrics, args, step=epoch) 
        
        history['losses'].append(avg_loss)
        history['epoch_times'].append(epoch_time)
        
        if is_main_process():
            print(f"Epoch [{epoch+1}/{num_epochs}] terminée")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Estimated remaining: {epoch_time * (num_epochs - epoch - 1):.1f}s")

        #### SAVE ####
        if is_main_process() and epoch+1 % args.save_every == 0:
            filepath = os.path.join(checkpoints_dir, f"simclr_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch+1, avg_loss, filepath)
            log_model_artifact(filepath, args, model_name=f"simclr_epoch_{epoch + 1}")
    
        if is_main_process():

            # Analyse périodique des représentations (tous les 10 epochs)
            if (epoch + 1) % 10 == 0:
                log_representation_quality(model, train_loader, args, f"epoch_{epoch+1}")

        if args.multi_gpu:
            train_sampler.set_epoch(epoch)


    total_time = time.time() - total_start_time
    if is_main_process():
        print(f"\nTraining terminé en {total_time:.1f}s")

        log_learning_curves(history, args, phase="training")
        log_performance_dashboard(history, args, phase="training")
    
    return history


def print_performance_summary(history):
    """
    Affiche un résumé détaillé des performances
    """
    print("\n" + "="*60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    
    # Temps moyens
    avg_batch_time = sum(history['batch_times']) / len(history['batch_times'])
    avg_transform_time = sum(history['transform_times']) / len(history['transform_times'])
    avg_forward_time = sum(history['forward_times']) / len(history['forward_times'])
    avg_backward_time = sum(history['backward_times']) / len(history['backward_times'])
    
    print(f"Temps moyen par batch: {avg_batch_time:.3f}s")
    print(f"Temps transformations: {avg_transform_time:.3f}s ({avg_transform_time/avg_batch_time*100:.1f}%)")
    print(f"Temps forward pass:    {avg_forward_time:.3f}s ({avg_forward_time/avg_batch_time*100:.1f}%)")
    print(f"Temps backward pass:   {avg_backward_time:.3f}s ({avg_backward_time/avg_batch_time*100:.1f}%)")
    
    # Loss evolution
    print(f"\nLoss Evolution:")
    print(f"Initial: {history['losses'][0]:.4f}")
    print(f"Final:   {history['losses'][-1]:.4f}")
    print(f"Change:  {((history['losses'][-1] - history['losses'][0]) / history['losses'][0] * 100):+.1f}%")
    
    # Temps par époque
    total_epochs = len(history['epoch_times'])
    total_time = sum(history['epoch_times'])
    avg_epoch_time = total_time / total_epochs
    
    print(f"\nTemps par époch:")
    print(f"Moyenne: {avg_epoch_time:.1f}s")
    print(f"Total:   {total_time:.1f}s")


def save_checkpoint(model, optimizer, epoch, losses, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {losses[-1]:.4f}")
    return epoch

def compute_detailed_contrastive_metrics(z_i, z_j, temperature, args):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature
    pos_similarities = torch.diag(sim_matrix, batch_size).mean()
    neg_pos_similarities = torch.diag(sim_matrix, -batch_size).mean()
    avg_positive_sim = (pos_similarities + neg_pos_similarities) / 2
    mask = torch.eye(2 * batch_size).bool()
    negative_similarities = sim_matrix[~mask].mean()
    softmax_sims = torch.softmax(sim_matrix, dim=1)
    entropy = -(softmax_sims * torch.log(softmax_sims + 1e-8)).sum(dim=1).mean()
    return {
        'contrastive/avg_positive_similarity': avg_positive_sim.item(),
        'contrastive/avg_negative_similarity': negative_similarities.item(),
        'contrastive/similarity_gap': (avg_positive_sim - negative_similarities).item(),
        'contrastive/entropy': entropy.item()
    }
