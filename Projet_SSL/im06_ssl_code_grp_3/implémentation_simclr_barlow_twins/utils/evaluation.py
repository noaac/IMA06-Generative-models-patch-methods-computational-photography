import torch
from tqdm import tqdm
import sys
import os
import time
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wandb_utils import log_metrics, log_confusion_matrix, log_learning_curves
from utils.multi_gpu_utils import is_main_process, gather_tensor_across_processes
import torch.distributed as dist
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import wandb



def retrain_and_evaluate_simclr(model, train_loader, test_loader, optimizer, args, train_sampler, val_sampler):
    """
    Fonction de réentraînement et d'évaluation de SimCLR
    """
    if is_main_process():
        print(f"Device: {args.device}")
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    ############ RE-ENTRAINEMENT #######################
    model.train()

    if args.multi_gpu:
        model.module.evaluation = True
    else:
        model.evaluation = True

    train_losses = []
    train_accuracies = [] 


    total_start_time = time.time()
    
    for epoch in range(args.eval_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        if is_main_process():
            print(f"\nEpoch [{epoch+1}/{args.eval_epochs}]")
            print("-" * 50)
        
        total = 0
        correct = 0
        
        for batch_idx, (x, labels) in enumerate(tqdm(train_loader, desc="Training")):
            batch_start_time = time.time()
            
            x = x.to(args.device)
            z = model(x)
            loss = criterion(z, labels.view(labels.shape[0]).to(args.device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_time = time.time() - batch_start_time

            _, predicted = torch.max(z.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(args.device)).sum().item()
            
            if batch_idx % args.log_interval_wandb == 0:
                batch_acc = 100.0 * correct / total if total > 0 else 0.0
                log_metrics({
                    'retrain_batch_loss': loss.item(),
                    'retrain_batch_accuracy': batch_acc,
                    'retrain_epoch': epoch,
                    'retrain_batch': batch_idx,
                }, args, step=epoch * len(train_loader) + batch_idx) 
            
            if is_main_process() and batch_idx % args.log_interval == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Total: {batch_time:.3f}s")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0
        
        train_losses.append(avg_loss)
        train_accuracies.append(epoch_acc)
        if is_main_process():
            log_metrics({
                'retrain_epoch_loss': avg_loss,
                'retrain_epoch_accuracy': epoch_acc,
                'retrain_epoch': epoch,
                'retrain_epoch_time': epoch_time,
            }, args, step=epoch)

            print(f"Epoch [{epoch+1}/{args.eval_epochs}] terminée")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Estimated remaining: {epoch_time * (args.eval_epochs - epoch - 1):.1f}s")

        if args.multi_gpu:
            train_sampler.set_epoch(epoch)
    
    total_time = time.time() - total_start_time

    dist.barrier()
    
    if is_main_process():
        print(f"\nTraining terminé en {total_time:.1f}s")

    ############ EVALUATION #######################
    if is_main_process():
        print(f"\nEvaluation")
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(tqdm(test_loader, desc="Evaluation", disable=not is_main_process())):

            x = x.to(args.device)
            labels = labels.to(args.device)
            z = model(x)

            loss = criterion(z, labels.view(labels.shape[0]).to(args.device))
            preds = z.softmax(dim=1).argmax(dim=1)

            # Synchronisation multi-GPU
            preds = gather_tensor_across_processes(preds)
            targets = gather_tensor_across_processes(labels)
            loss_tensor = gather_tensor_across_processes(loss.unsqueeze(0))


            all_preds.append(preds)
            all_targets.append(targets)
            all_losses.append(loss_tensor)
    
        # Concaténer tout
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets).view(-1)
        all_losses = torch.cat(all_losses)

        # Calcul des métriques uniquement sur le process principal
        if is_main_process():
            avg_eval_loss = all_losses.mean().item()
            print(all_preds.shape, all_targets.shape)
            accuracy = multiclass_accuracy(all_preds, all_targets, num_classes=args.num_classes)
            f1 = multiclass_f1_score(all_preds, all_targets, num_classes=args.num_classes)

            log_metrics({
                'final_test_accuracy': float(accuracy),
                'final_test_f1': float(f1),
                'final_test_loss': avg_eval_loss,
                'total_retrain_time': total_time,
            }, args)

            log_confusion_matrix(all_targets.tolist(), all_preds.tolist(), args, class_names=[str(i) for i in range(args.num_classes)])
        
            history = {
                'losses': train_losses,
                'accuracies': train_accuracies,
            }
            log_learning_curves(history, args, phase="retraining")

            metrics = {}
            metrics['accuracy'] = accuracy
            metrics['f1score'] = f1
            print(f"\nEvaluation terminée | Accuracy {metrics['accuracy']} | F1 Score {metrics['f1score']}")
    
    if is_main_process():
        log_detailed_evaluation_metrics(model, test_loader, args)

    if args.multi_gpu:
        model.module.evaluation = False
    else:
        model.evaluation = False
    
    return None


def log_detailed_evaluation_metrics(model, test_loader, args):
    # Métriques d'évaluation détaillées par classe
    if not args.use_wandb:return
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    # Métriques par classe
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0)
    class_metrics = {}
    for i in range(len(precision)):
        class_metrics.update({
            f"eval/class_{i}_precision": precision[i],
            f"eval/class_{i}_recall": recall[i],
            f"eval/class_{i}_f1": f1[i],
            f"eval/class_{i}_support": int(support[i])})
    # Métriques moyennes
    class_metrics.update({
        "eval/macro_precision": precision.mean(),
        "eval/macro_recall": recall.mean(),
        "eval/macro_f1": f1.mean(),
        "eval/weighted_f1": np.average(f1, weights=support)})
    wandb.log(class_metrics)
    # Table des résultats par classe
    class_names = [f"Class_{i}" for i in range(len(precision))]
    results_table = wandb.Table(
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
        data=[[name, float(p), float(r), float(f), int(s)] for name, p, r, f, s in 
              zip(class_names, precision, recall, f1, support)])
    wandb.log({"eval/per_class_metrics": results_table})
