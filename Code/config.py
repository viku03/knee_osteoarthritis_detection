from pathlib import Path
import torch
from datetime import datetime
import os
import numpy as np


class Config:
    def __init__(self):
        # Basic settings
        self.seed = 42
        self.num_classes = 5
        self.device = self._get_device()
        
        # Data settings
        self.train_dir = Path("/Users/Viku/Datasets/Medical/Knee")
        self.val_dir = Path("/Users/Viku/Datasets/Medical/Knee")
        self.output_dir = Path(f'model_outputs_v2/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.model_dir = self.output_dir / 'models'
        self._create_directories()

        # Training parameters
        self.num_epochs = 100
        self.num_folds = 5
        self.val_freq = 1
        self.early_stopping_patience = 15
        
        # Model parameters
        self.transformer_dim = 2048
        self.dropout = 0.4

        # Progressive image sizing
        self.size_schedule = [
            (0, 160),    # Start with 160x160
            (15, 192),   # Increase to 192x192 at epoch 15
            (30, 224)    # Final size 224x224 at epoch 30
        ]
        
        # Batch sizing
        self.batch_size = 32
        self.adaptive_batch_size = True
        self.batch_size_scaler = 1.05  # Reduce to 10% increase
        self.max_batch_size = 128    # Add a maximum limit
        self.min_improvement = 0.02  # Only increase if improvement seen
        self.gradient_accumulation_steps = 2

        # Optimizer settings
        self.max_lr = 2e-4
        self.min_lr = 1e-6
        self.weight_decay = 0.02
        self.max_grad_norm = 0.5
        self.div_factor = 25  # max_lr/div_factor = initial lr
        self.final_div_factor = 1e4
        self.initial_temp = 1.0

        
        # Learning rate schedule
        self.warmup_epochs = 8
        self.lr_scheduler = 'one_cycle'  # Options: 'cosine', 'one_cycle', 'plateau'
        self.scheduler_type = 'one_cycle'
        self.pct_start = 0.3  # 30% warmup in one cycle
        
        # Loss weights
        self.seg_weight = 0.4
        self.cls_weight = 0.6
        self.label_smoothing = 0.15


        # Dynamic class weights
        self.update_class_weights()

        # Enhanced sampling strategy
        self.use_weighted_sampler = True
        self.sampler_beta = 0.9995  # For effective number sampling
        
        # Augmentation settings
        self.mixup_prob = 0.5
        self.cutmix_prob = 0.3
        self.mosaic_prob = 0.2
        
        # Additional regularization
        self.label_smoothing = 0.1        
        
        # Modified learning rates
        self.cos_decay_epochs = 5



        # Regularization
        self.mixup_alpha = 0.4
        self.cutmix_alpha = 1.0
        self.random_erase_prob = 0.25
        self.ema_decay = 0.9997
        self.ema_warmup_epochs = 5

        # Performance optimization
        self.num_workers = 0
        self.pin_memory = False
        self.prefetch_factor = 4
        self.persistent_workers = True
        self.mixed_precision = True
        self.use_amp = True
        self.gradient_clip_val = 0.5

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _create_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.size_schedule[-1][1] <= 1024, "Invalid final image size"
        assert 0 < self.batch_size <= 512, "Invalid batch size"
        assert 0 < self.num_epochs <= 1000, "Invalid number of epochs"
        assert self.lr_scheduler in ['cosine', 'one_cycle', 'plateau'], "Invalid scheduler type"
    def update_class_weights(self):
        """Dynamic class weight calculation with temperature scaling and numerical stability"""
        class_counts = np.array([2286, 1046, 1516, 757, 173], dtype=np.float32)
        total_samples = np.sum(class_counts)
        temperature = 0.05  # Controls the smoothing of weights
        
        # Calculate effective numbers
        beta = 0.9999
        effective_numbers = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
        
        # Calculate initial weights
        weights = total_samples / (self.num_classes * effective_numbers)
        
        # Apply temperature scaling with numerical stability
        # First normalize weights to prevent overflow
        weights_norm = weights - np.max(weights)  # Subtract maximum for numerical stability
        weights = weights_norm / temperature
        
        # Compute softmax with numerical stability
        weights_exp = np.exp(weights - np.max(weights))  # Subtract maximum again for numerical stability
        weights = weights_exp / np.sum(weights_exp)
        
        # Convert to torch tensor
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)


config = Config()