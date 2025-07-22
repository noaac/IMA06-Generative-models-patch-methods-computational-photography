import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import kornia
import kornia.augmentation as K

# 1. TRANSFORMATIONS NAIVES

def RandomCropUniform(image):
    """
    Fait un crop aléatoire de l'image. 
    J'ai choisi de faire des crops de taille comprise entre 30% et 70% de la taille originale.
    """
    h,w = image.size[0],image.size[1]
    new_h = np.random.uniform(low=0.3*h, high=0.7*h)
    new_w = np.random.uniform(low=0.3*w, high=0.7*w)
    h_start = np.random.randint(low=0, high=h-new_h)
    w_start = np.random.randint(low=0, high=w-new_w)
    image = image.crop((w_start, h_start, w_start+new_w, h_start+new_h))
    return image


def RandomColor(image):
    """
    Fait un changement aléatoire de la couleur de l'image.
    """
    color_jitter = transforms.ColorJitter(
        brightness=np.random.uniform(0.5, 1.5),
        contrast=np.random.uniform(0.5, 1.5),
        saturation=np.random.uniform(0.5, 1.5)
    )
    return color_jitter(image)

def RandomGaussianBlur(image):
    """
    Applique un flou gaussien aléatoire à l'image.
    """
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=23,
        sigma=np.random.uniform(0.1, 2.0)
    )
    return gaussian_blur(image)


def pipeline_transformations_1():
    """
    Pipeline de transformations original de votre notebook
    Fonctionne sur des images PIL
    """
    return transforms.Compose([
        RandomCropUniform,
        RandomColor, 
        RandomGaussianBlur,
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def apply_transforms_1(batch, transform_pipeline):
    """
    Applique les transformations sur un batch complet
    """
    return torch.stack([transform_pipeline(image) for image in batch])


def pipeline_transforms_kornia(image_size=(28, 28),crop=True, color=True, blur=True,Rotation=False):
    transforms = []
    if crop :
        transforms.append(K.RandomResizedCrop(
            size=(image_size[0], image_size[1]),
            scale=(0.3, 0.7),
            ratio=(0.75, 1.33),
            p=1.0
        ))
    if color :
        transforms.append(K.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.2, 0.2),
            p=0.9
        )) 
    if blur : 
        transforms.append(K.RandomGaussianBlur(
            kernel_size=(23, 23),
            sigma=(0.1, 2.0),
            p=0.8
        ))
    else : 
        transforms.append(KF.Sobel(
            normalized = True)
        )
    if Rotation :
        transforms.append(K.RandomRotation(
            degrees=(-30, 30),
            same_on_batch=True
        ))
    K.Normalize(mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
    return nn.Sequential(*transforms)
    
def apply_transforms_kornia(batch, transform_pipeline):
    return transform_pipeline(batch)




def get_simclr_transforms(use_gpu=True, image_size=28, args=None):
    if use_gpu and args: return pipeline_transforms_kornia(
        image_size=(image_size, image_size),
        crop=args.transform_crop,
        color=args.transform_color,
        blur=args.transform_blur,
        Rotation=args.transform_rotation
        )
    elif use_gpu: return pipeline_transforms_kornia(image_size=(image_size, image_size))
    else: return pipeline_transformations_1()


def apply_transforms_to_batch(batch, transform_pipeline, use_gpu=False):
    if use_gpu: return apply_transforms_kornia(batch, transform_pipeline)
    else: return apply_transforms_1(batch, transform_pipeline)