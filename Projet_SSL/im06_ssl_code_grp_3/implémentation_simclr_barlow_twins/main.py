import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim
import argparse
from models.simclr import create_simclr_model
from data.dataset import get_pathmnist_loaders, get_pathmnist_evaluation_loaders
from data.transforms import get_simclr_transforms
from utils.train import train_simclr, print_performance_summary, save_checkpoint
from utils.evaluation import retrain_and_evaluate_simclr
from config import print_config, set_seed
from lars_optim import lars
from utils.train import load_checkpoint

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.utils.data.distributed as dist_data
import torch.distributed as dist
from utils.multi_gpu_utils import setup_ddp, cleanup_ddp, is_main_process

from utils.wandb_utils import init_wandb, finish_wandb, log_metrics, log_hyperparameter_impact

from models.barlow_twins import create_BT_model


def main():

    set_seed()

    ########## FILE HANDLING #######
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='SimCLR Training Configuration')

    # Experiment Setup
    parser.add_argument('--experiment_name', type=str, default='test_new_wandb', help='Name of the experiment')
    parser.add_argument('--train', type=bool, default=True, help='Run training phase')
    parser.add_argument('--evaluate', type=bool, default=True, help='Run evaluation phase')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to resume checkpoint')
    
    # Model
    parser.add_argument('--SimClr', type=bool, default=True, help='use SimClr or Barlow Twins if False')
    
    # ResNet Encoder
    parser.add_argument('--encoder', type=str, default='resnet50', help='Encoder backbone (e.g., resnet18, resnet50)')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained weights for encoder')

    # Projection Head
    parser.add_argument('--proj_input_dim', type=int, default=2048, help='Projection head input dimension')
    parser.add_argument('--proj_hidden_dim', type=int, default=512, help='Projection head hidden layer dimension')
    parser.add_argument('--proj_output_dim', type=int, default=128, help='Projection head output dimension')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pin_memory', type=bool, default=False, help='Use pin memory in DataLoader')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate, auto computed if none')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for contrastive loss')
    parser.add_argument('--lambda_param', type=float, default=5e-3, help='Lambda parameter for Barlow Twins Loss')

    # Warmup & LR Scheduling
    parser.add_argument('--warmup', type=bool, default=True, help='Enable warmup')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--cosine_decay', type=bool, default=True, help='Use cosine learning rate decay')

    # Optimization
    parser.add_argument('--use_lars', type=bool, default=True, help='Use LARS optimizer')
    parser.add_argument('--vectorized_loss', type=bool, default=True, help='Use vectorized loss')
    parser.add_argument('--gpu_transforms', type=bool, default=True, help='Use GPU transforms instead of CPU')

    # Evaluation
    parser.add_argument('--eval_epochs', type=int, default=50, help='Number of epochs for linear evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Evaluation batch size')
    parser.add_argument('--eval_lr', type=float, default=1e-3, help='Evaluation learning rate')
    parser.add_argument('--eval_weight_decay', type=float, default=1e-6, help='Evaluation weight decay')
    parser.add_argument('--cut_ratio', type=float, default=0.2, help='Ratio of trainset used for evaluation training')
    parser.add_argument('--eval_from_ckpt', type=bool, default=False, help='Evaluate from checkpoint')
 
    # Logging & Checkpointing
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval (in batches)')
    parser.add_argument('--log_interval_wandb', type=int, default=50, help='Logging interval (in batches) for Wandb')
    parser.add_argument('--save_every', type=int, default=20, help='Save checkpoint every N epochs')

    # Dataset
    parser.add_argument('--dataset', type=str, default='pathmnist', help='Dataset name')
    parser.add_argument('--image_size', type=int, default=128, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of target classes')
    parser.add_argument('--download', type=bool, default=True, help='Download dataset if not found locally')

    # Batch norm sync (multi-GPU)
    parser.add_argument('--sync_batch_norm', type=bool, default=True, help='Enable Sync BatchNorm')
    parser.add_argument('--multi_gpu', type=bool, default=True, help='Enable multi gpu')

    # Device
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", help='Device')

    # Wandb
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use Wandb api')
    parser.add_argument('--wandb_mode', type=str, default='offline', help='Mode, offline or online')
    parser.add_argument('--wandb_project', type=str, default='logs-test', help='Project')
    parser.add_argument('--wandb_entity', type=str, default='ssl-im06', help='Entity')
    parser.add_argument('--wandb_dir', type=str, default=os.path.join(PROJECT_ROOT, "wandb"), help='Directory where wandb logs are saved')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional run name for wandb')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=["simclr", "pathmnist", "ssl"], help='List of tags for wandb run')
    parser.add_argument('--wandb_log_interval', type=int, default=1, help='Logging interval (in batches)')
    parser.add_argument('--wandb_save_model', type=bool, default=True, help='Save model with wandb')
    
    parser.add_argument('--transform_crop', type=bool, default='true',help='Utiliser ou non RandomResizedCrop dans les transformations (True/False)')
    parser.add_argument('--transform_color', type=bool, default='true',help='Utiliser ou non ColorJitter dans les transformations (True/False)')
    parser.add_argument('--transform_blur', type=bool, default='true',help='Utiliser ou non RandomGaussianBlur dans les transformations (True/False)')
    parser.add_argument('--transform_rotation', type=bool, default='true',help='Utiliser ou non RandomRotation dans les transformations (True/False)')

    args = parser.parse_args()
    
    ########## FILES HANDLING #######
    
    CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints_" + args.experiment_name)
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs_" + args.experiment_name)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if args.lr is None:
        args.lr = 0.3 * args.batch_size / 256

    os.environ['WANDB_SILENT'] = 'true'
    if is_main_process():
        wandb_run = init_wandb(args)
        log_hyperparameter_impact(args)

    ########## TRAINING ############

    try:

        if args.train:
            
            if is_main_process():
                # Afficher la configuration
                print_config(args)
                if args.SimClr:
                    print(f"Starting SimCLR training...")
                else :
                    print(f"Starting BarlowTwins training...")
                print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
                print(f"GPU transforms: {args.gpu_transforms}, Vectorized loss: {args.vectorized_loss}")
            
            # print("\nCreating model...")
            if args.SimClr:
                model = create_simclr_model(args)
            else :
                model = create_BT_model(args)

            if args.multi_gpu:
                setup_ddp()
                local_rank = int(os.environ["LOCAL_RANK"])
                args.device = torch.device(f"cuda:{local_rank}")
                model = model.to(args.device)
                model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

            else:
                model = model.to(args.device)

            
            print(f"Loading Data")
            train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_pathmnist_loaders(args)
            
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Transformations
            transform_pipeline = get_simclr_transforms(use_gpu=args.gpu_transforms, image_size=args.image_size, args=args)
            if args.gpu_transforms and hasattr(transform_pipeline, 'to'): transform_pipeline = transform_pipeline.to(args.device)
            
            if args.use_lars:
                optimizer = lars.Lars(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            start_epoch = 0
            
            if args.checkpoint:
                start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
                num_epochs = args.epochs - start_epoch
                print(f" Resuming from epoch {start_epoch}")
            else:
                num_epochs = args.epochs
            
            print("\nStarting training...")
            
            try:
                history = train_simclr(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    transform_pipeline=transform_pipeline,
                    num_epochs=num_epochs,
                    args=args,
                    checkpoints_dir=CHECKPOINTS_DIR,
                    logs_dir=LOGS_DIR,
                    train_sampler=train_sampler)
                
                if is_main_process():
                    print_performance_summary(history)
                    
                    # Sauvegarde finale
                    final_checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"simclr_epoch_{start_epoch + num_epochs}_final.pth")
                    save_checkpoint(model=model,
                                    optimizer=optimizer,
                                    epoch=start_epoch + num_epochs,
                                    losses=history['losses'],
                                    filepath=final_checkpoint_path)

                    
                    print(f"\nTraining completed successfully!")
                    print(f"Final model saved to: {final_checkpoint_path}")
                
            except KeyboardInterrupt:
                if is_main_process():
                    print("\nTraining interrupted by user")
                    emergency_path = os.path.join(CHECKPOINTS_DIR, f"simclr_epoch_{start_epoch + num_epochs}_interrupted.pth")
                    save_checkpoint(model, optimizer, start_epoch, 0.0, emergency_path)
                    print(f"Emergency checkpoint saved to: {emergency_path}")

                args.evaluate = False
            
            except Exception as e:
                args.evaluate = False
                
                print(f"\nTraining failed with error: {e}")
                raise

            # if args.multi_gpu:
            #     cleanup_ddp()

        ########## EVALUATION ############

        if args.evaluate:

            dist.barrier()

            if args.eval_from_ckpt:
                if args.SimClr:
                    model = create_simclr_model(args)
                else :
                    model = create_BT_model(args)
                model = model.to(args.device)
                _ = load_checkpoint(args.checkpoint, model)

                print(f" Loading model epoch {args.checkpoint}")
                if not args.train:
                    print_config(args)
            
            # if args.multi_gpu:

                # setup_ddp()
                # local_rank = int(os.environ["LOCAL_RANK"])
                # args.device = torch.device(f"cuda:{local_rank}")

                # if isinstance(model, DDP):
                #     model = model.module
                #     print('HERE')

            #     model = model.to(args.device)
            #     model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
            # else:
            #     model = model.to(args.device)

            if is_main_process():
                print("\nStarting Re-training and evaluation...")

            cut_train_loader, val_loader, cut_train_sampler, val_sampler = get_pathmnist_evaluation_loaders(args)
            training_optimizer = optim.Adam(model.parameters(), lr=args.eval_lr, weight_decay=args.eval_weight_decay)

            try:
                metrics = retrain_and_evaluate_simclr(model=model,
                                                    train_loader=cut_train_loader,
                                                    test_loader=val_loader,
                                                    optimizer=training_optimizer,
                                                    args=args,
                                                    train_sampler=cut_train_sampler,
                                                    val_sampler=val_sampler)
                
                if is_main_process():
                    print(f"\nEvaluation completed successfully!")

                    log_metrics({'evaluation_completed': True}, args)

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
            
            except Exception as e:
                print(f"\nTraining failed with error: {e}")
                log_metrics({'evaluation_failed': True, 'error': str(e)}, args)
                raise

            # if args.multi_gpu:
            #     cleanup_ddp()
    
    finally:
        if is_main_process():
             finish_wandb(args)

        if args.multi_gpu:
            cleanup_ddp()


if __name__ == "__main__":
    main()