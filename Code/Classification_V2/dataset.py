from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast
from PIL import Image
import os
import random
import torch
import numpy as np
from torch.utils.data import Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Optional, Dict, List, Iterator
from collections import Counter


class AdvancedXRayTransforms:
    @staticmethod
    def get_train_transform():
        return A.Compose([
            # Careful geometric transforms for knee alignment
            A.RandomResizedCrop(
                224, 224,
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                p=1.0
            ),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            
            # X-ray specific enhancements
            A.OneOf([
                A.GaussNoise(var_limit=(5, 20), p=1.0),
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            ], p=0.3),
            
            # Enhance bone details
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.5
            ),
            
            # Contrast adjustment for better bone visibility
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            
            # X-ray specific normalization
            A.Normalize(
                mean=[0.0],  # Changed to 0 for grayscale
                std=[1.0],   # Keep unit variance
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transform():
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase='train', mixup_alpha=0.2):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.mixup_alpha = mixup_alpha if phase == 'train' else 0
        self.image_paths = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        for class_idx in range(5):
            class_dir = os.path.join(self.root_dir, str(class_idx))
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in images:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def apply_mixup(self, image1, image2, label1, label2, alpha):
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        return mixed_image, label1, label2, lam

    def __getitem__(self, idx):
        # Load primary image
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('L'))
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Always return consistent output size (image, target_a, target_b, lam)
        if self.mixup_alpha > 0 and self.phase == 'train' and random.random() < 0.5:
            mix_idx = random.randint(0, len(self) - 1)
            mix_img_path = self.image_paths[mix_idx]
            mix_image = np.array(Image.open(mix_img_path).convert('L'))
            mix_label = self.labels[mix_idx]
            
            if self.transform:
                transformed_mix = self.transform(image=mix_image)
                mix_image = transformed_mix['image']
            
            mixed_image, label_a, label_b, lam = self.apply_mixup(
                image, mix_image,
                label, mix_label,
                self.mixup_alpha
            )
            return mixed_image, label_a, label_b, lam
        
        # For non-mixup cases, return the same size output with dummy values
        return image, label, label, 1.0  # lam = 1.0 means no mixup

    def __len__(self):
        return len(self.image_paths)

def custom_collate(batch):
    """Custom collate function to handle both mixed-up and regular batches"""
    images, targets_a, targets_b, lams = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Convert targets to tensors
    targets_a = torch.tensor(targets_a, dtype=torch.long)
    targets_b = torch.tensor(targets_b, dtype=torch.long)
    lams = torch.tensor(lams, dtype=torch.float)
    
    return images, targets_a, targets_b, lams

class AdvancedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        balance_strategy: str = 'oversample',  # 'oversample', 'weights', or 'stratified'
        oversample_multiplier: float = 1.2,
        dynamic_balance: bool = False,
        min_class_samples: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Advanced batch sampler with multiple strategies for handling imbalanced datasets.
        
        Args:
            dataset: Dataset with labels attribute
            batch_size: Number of samples per batch
            balance_strategy: Strategy for handling class imbalance
            oversample_multiplier: Multiplier for oversampling (only used if balance_strategy='oversample')
            dynamic_balance: Whether to adjust sampling weights during training
            min_class_samples: Minimum samples per class in each batch
            random_state: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.balance_strategy = balance_strategy
        self.oversample_multiplier = oversample_multiplier
        self.dynamic_balance = dynamic_balance
        self.min_class_samples = min_class_samples
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Group indices by class
        self.indices_per_class: Dict[int, List[int]] = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.indices_per_class:
                self.indices_per_class[label] = []
            self.indices_per_class[label].append(idx)

        # Print initial distribution
        self._print_class_distribution()
        
        # Initialize sampling parameters
        self._initialize_sampling_parameters()
        
        # Verify batch size compatibility
        self._verify_batch_size()
        
        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1

        # Initialize dynamic balance tracking
        if self.dynamic_balance:
            self.sampled_counts = Counter()

    def _print_class_distribution(self) -> None:
        """Print initial class distribution statistics"""
        print("\nInitial class distribution:")
        total_samples = sum(len(indices) for indices in self.indices_per_class.values())
        
        for cls, indices in sorted(self.indices_per_class.items()):
            count = len(indices)
            percentage = (count / total_samples) * 100
            print(f"Class {cls}: {count} samples ({percentage:.1f}%)")

    def _initialize_sampling_parameters(self) -> None:
        """Initialize sampling weights and probabilities based on chosen strategy"""
        total_samples = sum(len(indices) for indices in self.indices_per_class.values())
        num_classes = len(self.indices_per_class)
        target_fraction = 1.0 / num_classes

        self.weights = {}
        self.sampling_probs = {}

        for cls, indices in self.indices_per_class.items():
            class_fraction = len(indices) / total_samples

            if self.balance_strategy == 'oversample':
                if class_fraction < target_fraction:
                    self.weights[cls] = (target_fraction / class_fraction) * self.oversample_multiplier
                else:
                    self.weights[cls] = 1.0
                    
            elif self.balance_strategy == 'weights':
                # Inverse frequency weighting
                self.weights[cls] = 1.0 / class_fraction
                
            elif self.balance_strategy == 'stratified':
                # Equal probability for all classes
                self.weights[cls] = 1.0
                self.sampling_probs[cls] = target_fraction
                
        # Normalize weights if using weight-based sampling
        if self.balance_strategy == 'weights':
            total_weight = sum(self.weights.values())
            for cls in self.weights:
                self.weights[cls] /= total_weight

        print(f"\nInitialized with {self.balance_strategy} strategy:")
        for cls, weight in sorted(self.weights.items()):
            print(f"Class {cls}: weight = {weight:.2f}")

    def _verify_batch_size(self) -> None:
        """Verify batch size compatibility and print warnings if needed"""
        if self.batch_size < len(self.indices_per_class):
            print("\nWarning: batch_size is smaller than number of classes")
            print("This might lead to underrepresentation of some classes")
            print(f"Consider increasing batch size (current: {self.batch_size})")
            
        if self.min_class_samples and self.min_class_samples * len(self.indices_per_class) > self.batch_size:
            print("\nWarning: minimum class samples constraint cannot be satisfied")
            print(f"Required: {self.min_class_samples * len(self.indices_per_class)}")
            print(f"Available: {self.batch_size}")

    def _get_sample_pools(self) -> Dict[int, List[int]]:
        """Create sampling pools based on chosen strategy"""
        class_pools = {}
        
        for cls, indices in self.indices_per_class.items():
            pool = []
            
            if self.balance_strategy == 'oversample':
                # Oversample minority classes
                multiplier = int(self.weights[cls] * 2)  # Double for more aggressive oversampling
                pool.extend(indices * multiplier)
                
                # Add additional random samples if needed
                remaining = int((self.weights[cls] * 2 - multiplier) * len(indices))
                if remaining > 0:
                    pool.extend(random.sample(indices, remaining))
                    
            else:  # 'weights' or 'stratified'
                pool.extend(indices)
                
            random.shuffle(pool)
            class_pools[cls] = pool
            
        return class_pools

    def _update_dynamic_weights(self) -> None:
        """Update weights based on sampling history"""
        if not self.dynamic_balance:
            return
            
        total_samples = sum(self.sampled_counts.values())
        if total_samples == 0:
            return
            
        target_fraction = 1.0 / len(self.indices_per_class)
        
        for cls in self.indices_per_class:
            current_fraction = self.sampled_counts[cls] / total_samples
            if current_fraction < target_fraction:
                self.weights[cls] *= 1.1  # Gradually increase weight
            elif current_fraction > target_fraction:
                self.weights[cls] *= 0.9  # Gradually decrease weight

    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches according to chosen strategy"""
        class_pools = self._get_sample_pools()
        
        for _ in range(self.num_batches):
            batch = []
            
            # Ensure minimum samples per class if specified
            if self.min_class_samples:
                for cls in self.indices_per_class:
                    if len(class_pools[cls]) < self.min_class_samples:
                        class_pools[cls] = self.indices_per_class[cls].copy()
                        random.shuffle(class_pools[cls])
                    
                    samples = class_pools[cls][:self.min_class_samples]
                    batch.extend(samples)
                    class_pools[cls] = class_pools[cls][self.min_class_samples:]
            
            # Fill remaining slots based on strategy
            remaining_slots = self.batch_size - len(batch)
            
            if self.balance_strategy == 'stratified':
                # Sample equally from all classes
                samples_per_class = remaining_slots // len(self.indices_per_class)
                for cls in self.indices_per_class:
                    if len(class_pools[cls]) < samples_per_class:
                        class_pools[cls] = self.indices_per_class[cls].copy()
                        random.shuffle(class_pools[cls])
                    batch.extend(class_pools[cls][:samples_per_class])
                    class_pools[cls] = class_pools[cls][samples_per_class:]
                    
            else:  # 'oversample' or 'weights'
                # Sample based on weights
                all_pools = []
                for cls, pool in class_pools.items():
                    pool_contribution = int(remaining_slots * self.weights[cls])
                    all_pools.extend(pool[:pool_contribution])
                
                if len(all_pools) > 0:
                    random.shuffle(all_pools)
                    batch.extend(all_pools[:remaining_slots])
            
            # Track sampled classes if using dynamic balancing
            if self.dynamic_balance:
                self.sampled_counts.update(self.dataset.labels[i] for i in batch)
                self._update_dynamic_weights()
            
            # Final shuffle
            random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        return self.num_batches