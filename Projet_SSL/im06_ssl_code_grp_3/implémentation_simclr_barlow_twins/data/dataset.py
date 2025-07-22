import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.transforms import get_simclr_transforms
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(0)

def get_pathmnist_loaders(args):
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    if args.gpu_transforms:
        basic_transform = transforms.Compose([transforms.ToTensor(),])
    else:
        basic_transform = None
    
    train_dataset = DataClass(split='train', transform=basic_transform, download=args.download)
    val_dataset = DataClass(split='val', transform=basic_transform, download=args.download)
    test_dataset = DataClass(split='test', transform=basic_transform, download=args.download)
    
    if args.multi_gpu:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler   = DistributedSampler(val_dataset, shuffle=False)
            test_sampler  = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler

def get_pathmnist_evaluation_loaders(args):
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    if args.gpu_transforms:
        basic_transform = transforms.Compose([transforms.ToTensor(),])
    else:
        basic_transform = None
    
    train_dataset = DataClass(split='train', transform=basic_transform, download=args.download)
    cut_indices = torch.randperm(len(train_dataset))[:int(len(train_dataset)*args.cut_ratio)]
    cut_train_dataset = torch.utils.data.Subset(train_dataset, indices=cut_indices)
    val_dataset = DataClass(split='val', transform=basic_transform, download=args.download)

    if args.multi_gpu:
            cut_train_sampler = DistributedSampler(cut_train_dataset, shuffle=True)
            val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    else:
        cut_train_sampler = None
        val_sampler = None
    

    cut_train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(cut_train_sampler is None),
        sampler=cut_train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    return cut_train_loader, val_loader, cut_train_sampler, val_sampler



def get_dataset_info(data_flag='pathmnist'):
    info = INFO[data_flag]
    return {
        'num_classes': len(info['label']),
        'image_size': (28, 28),
        'channels': 3,
        'task': info['task'],
        'n_samples': info['n_samples']
    }