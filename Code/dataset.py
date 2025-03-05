import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any
import cv2
from multiprocessing import Lock
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from augmentation import AdvancedAugmentation, MixUpAugmentation
from config import config


@lru_cache(maxsize=2000)
def cached_load_image(path: str, img_size: int) -> np.ndarray:
    """Load and process image with caching."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize immediately after loading to save memory
    if img.shape[0] > img_size or img.shape[1] > img_size:
        scale = img_size / max(img.shape[0], img.shape[1])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    return img

class KneeDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 phase: str = 'train',
                 transform: Any = None,
                 img_size: int = 160,
                 enable_cache: bool = True,
                 cache_size: int = 2000):
        """
        Initialize dataset with explicit type conversion for statistics
        """
        # Input validation
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist")
        if phase not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid phase: {phase}")
        if img_size <= 0:
            raise ValueError("Image size must be positive")

        self.root_dir = os.path.join(root_dir, phase)
        self.phase = phase
        self.img_size = img_size
        self.enable_cache = enable_cache
        self.cache_size = cache_size

        # Initialize augmentations based on phase
        if self.phase == 'train':
            self.advanced_augmentation = AdvancedAugmentation(config)
            self.mixup = MixUpAugmentation(alpha=0.1)
        else:
            self.advanced_augmentation = None
            self.mixup = None
        
        # Load samples and calculate stats
        self.samples = self._load_samples()
        self.class_counts = self._get_class_counts()
        
        # Get statistics
        self.mean, self.std = self._compute_statistics()
        
        # Set transforms
        self.transform = transform or self._default_transforms()
        
        # Initialize image cache as a dictionary if enabled
        if self.enable_cache:
            self.image_cache = {}
        
        print(f"Initialized {phase} dataset with {len(self.samples)} samples")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Class distribution: {self.class_counts}")
        print(f"Mean: {self.mean}")
        print(f"Std: {self.std}")

    def _load_samples(self) -> list:
        """Load all image paths and their corresponding grades."""
        samples = []
        for grade in range(5):
            grade_dir = os.path.join(self.root_dir, str(grade))
            if os.path.exists(grade_dir):
                for img_file in os.listdir(grade_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(grade_dir, img_file)
                        side = 'left' if 'L' in img_file else 'right' if 'R' in img_file else 'unknown'
                        samples.append({
                            'image_path': img_path,
                            'grade': grade,
                            'side': side
                        })
        return sorted(samples, key=lambda x: x['image_path'])

    def _get_class_counts(self) -> Dict[int, int]:
        """Count samples for each grade."""
        counts = {i: 0 for i in range(5)}
        for sample in self.samples:
            counts[sample['grade']] += 1
        return counts

    def _compute_statistics(self):
        """Compute dataset statistics for grayscale images."""
        stats_path = f"{self.root_dir}_stats.pt"
        
        try:
            if os.path.exists(stats_path):
                stats = torch.load(stats_path)
                mean = float(stats["mean"])
                std = float(stats["std"])
                return mean, std
        except Exception as e:
            print(f"Could not load existing statistics: {e}")
        
        def process_image(sample):
            img = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return img.astype(np.float32) / 255.0
        
        with ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(process_image, self.samples[:1000]))
        
        processed_images = [img for img in processed_images if img is not None]
        
        if not processed_images:
            return 0.449, 0.226  # Default values for grayscale
        
        stacked_images = np.stack(processed_images)
        mean = float(np.mean(stacked_images))
        std = float(np.std(stacked_images))
        
        torch.save({"mean": mean, "std": std}, stats_path)
        
        return mean, std

    def _default_transforms(self):
        """Set up default transforms based on phase."""
        if self.phase == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                # Use advanced augmentation transforms if in training phase
                *([self.advanced_augmentation.transform] if self.advanced_augmentation else []),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def __load_image(self, path: str) -> np.ndarray:
        """Load grayscale image and convert to RGB."""
        if self.enable_cache and path in self.image_cache:
            return self.image_cache[path]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        
        if img.shape[0] > self.img_size or img.shape[1] > self.img_size:
            scale = self.img_size / max(img.shape[0], img.shape[1])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.enable_cache:
            if len(self.image_cache) >= self.cache_size:
                self.image_cache.pop(next(iter(self.image_cache)))
            self.image_cache[path] = img
            
        return img

    def update_transforms(self, resolution: int):
        """Update transforms with new resolution."""
        self.img_size = resolution
        self.transform = self._default_transforms()
        if self.enable_cache:
            self.image_cache.clear()

    def get_sampling_weights(self) -> torch.Tensor:
        """Calculate class-balanced sampling weights."""
        weights = []
        total_samples = sum(self.class_counts.values())
        
        for sample in self.samples:
            class_weight = total_samples / (self.class_counts[sample["grade"]] * len(self.class_counts))
            weights.append(class_weight)
        
        return torch.DoubleTensor(weights)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with optional mixup augmentation."""
        sample = self.samples[idx]
        
        # Load and transform image
        image = self.__load_image(sample["image_path"])
        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # Create mask
        mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        
        # Prepare batch
        batch = {
            "image": image,
            "mask": mask,
            "grade": torch.tensor(sample["grade"], dtype=torch.long),
            "side": sample["side"],
            "path": sample["image_path"]
        }
        
        # Apply mixup if in training phase
        if self.phase == 'train' and self.mixup is not None:
            batch = self.mixup(batch)
        
        return batch

    def __len__(self) -> int:
        return len(self.samples)