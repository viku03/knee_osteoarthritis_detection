import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


class AdvancedAugmentation:
    def __init__(self, config):
        self.transform = A.Compose([
            # Geometric transformations - more conservative
            A.RandomRotate90(p=0.3),  # Reduced probability
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,      # Reduced shift
                scale_limit=0.1,       # Reduced scale
                rotate_limit=15,       # Reduced rotation
                p=0.3                  # Reduced probability
            ),
            
            # Noise and blur - gentler
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=1),  # Reduced variance
                A.GaussianBlur(blur_limit=(3, 5), p=1),    # Reduced blur
                A.ISONoise(intensity=(0.1, 0.3), p=1)      # Reduced intensity
            ], p=0.2),
            
            # Intensity transformations - more conservative
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,    # Reduced from default
                    contrast_limit=0.1,      # Reduced from default
                    p=1
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1),  # More conservative range
                A.CLAHE(clip_limit=2.0, p=1)                # Reduced clip limit
            ], p=0.3),
            
            # Reduced dropout
            A.CoarseDropout(
                max_holes=4,        # Reduced holes
                max_height=4,       # Reduced size
                max_width=4,        # Reduced size
                p=0.2              # Reduced probability
            )
        ])

class MixUpAugmentation:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
    def __call__(self, batch):
        if self.alpha <= 0:
            return batch
            
        # Check if we're dealing with a single sample or a batch
        is_single_sample = not isinstance(batch['image'], torch.Tensor) or len(batch['image'].shape) == 3
        
        if is_single_sample:
            # Convert single sample to batch format
            batch = {
                'image': batch['image'].unsqueeze(0),
                'mask': batch['mask'].unsqueeze(0),
                'grade': torch.tensor([batch['grade']], dtype=torch.long),
                'side': [batch['side']],
                'path': [batch['path']]
            }
            
        batch_size = batch['image'].size(0)
        
        # Only apply mixup if we have more than one sample
        if batch_size > 1:
            lam = np.random.beta(self.alpha, self.alpha)
            rand_idx = torch.randperm(batch_size)
            
            # Mix the images and masks
            mixed_images = lam * batch['image'] + (1 - lam) * batch['image'][rand_idx]
            mixed_masks = lam * batch['mask'] + (1 - lam) * batch['mask'][rand_idx]
            
            # Store mixed data
            batch['mixed'] = {
                'image': mixed_images,
                'mask': mixed_masks,
                'grade': batch['grade'],
                'grade_shuffled': batch['grade'][rand_idx],
                'lambda': lam
            }
        else:
            # For single samples, create a dummy mixed entry with original data
            batch['mixed'] = {
                'image': batch['image'],
                'mask': batch['mask'],
                'grade': batch['grade'],
                'grade_shuffled': batch['grade'],
                'lambda': 1.0
            }
        
        # If original input was a single sample, convert back to single sample format
        if is_single_sample:
            batch = {
                'image': batch['image'].squeeze(0),
                'mask': batch['mask'].squeeze(0),
                'grade': batch['grade'].item(),
                'side': batch['side'][0],
                'path': batch['path'][0],
                'mixed': {
                    'image': batch['mixed']['image'].squeeze(0),
                    'mask': batch['mixed']['mask'].squeeze(0),
                    'grade': batch['mixed']['grade'].item(),
                    'grade_shuffled': batch['mixed']['grade_shuffled'].item(),
                    'lambda': batch['mixed']['lambda']
                }
            }
            
        return batch
    
class AugmentationPipeline:
    def __init__(self, config):
        self.config = config
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.size_schedule[0][1],
                width=config.size_schedule[0][1],
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(
                height=config.size_schedule[-1][1],
                width=config.size_schedule[-1][1]
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])

class MixupCutmixCollator:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        # Apply Mixup or Cutmix with probabilities
        if random.random() < self.config.mixup_prob:
            return self.mixup(batch)
        elif random.random() < self.config.cutmix_prob:
            return self.cutmix(batch)
        
        # Default: Pass through without modifications
        return default_collate(batch)

    def mixup(self, batch):
        # Stack all images, grades, and preserve other keys
        images = torch.stack([item['image'] for item in batch])
        grades = torch.tensor([item['grade'] for item in batch])  # Labels to augment
        masks = torch.stack([item['mask'] for item in batch])  # Preserve mask
        sides = [item['side'] for item in batch]               # Preserve side
        paths = [item['path'] for item in batch]               # Preserve path

        # Generate mixup weights
        alpha = self.config.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        
        batch_size = len(batch)
        index = torch.randperm(batch_size)  # Random permutation for pairing
        
        # Create mixed images and labels
        mixed_images = lam * images + (1 - lam) * images[index]
        grade_a, grade_b = grades, grades[index]

        # Return mixed batch with all keys
        return {
            'image': mixed_images,
            'grade_a': grade_a,  # Mixed label A
            'grade_b': grade_b,  # Mixed label B
            'lam': lam,          # Lambda for weighting
            'mask': masks,       # Pass masks unchanged
            'side': sides,       # Pass sides unchanged
            'path': paths        # Pass paths unchanged
        }

    def cutmix(self, batch):
        # Stack all images, grades, and preserve other keys
        images = torch.stack([item['image'] for item in batch])
        grades = torch.tensor([item['grade'] for item in batch])  # Labels to augment
        masks = torch.stack([item['mask'] for item in batch])  # Preserve mask
        sides = [item['side'] for item in batch]               # Preserve side
        paths = [item['path'] for item in batch]               # Preserve path

        # Generate random box for cutmix
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        batch_size = len(batch)
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to match actual pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        grade_a, grade_b = grades, grades[index]

        # Return mixed batch with all keys
        return {
            'image': images,
            'grade_a': grade_a,  # Mixed label A
            'grade_b': grade_b,  # Mixed label B
            'lam': lam,          # Lambda for weighting
            'mask': masks,       # Pass masks unchanged
            'side': sides,       # Pass sides unchanged
            'path': paths        # Pass paths unchanged
        }

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
