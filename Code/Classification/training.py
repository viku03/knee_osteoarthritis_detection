import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data.sampler import WeightedRandomSampler
from pathlib import Path
import time
import json
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score
)
import math
from torch.optim.lr_scheduler import LambdaLR


from model_arch import HybridModel
from training_monitor import TrainingMonitor

class Config:
    # Paths
    data_root = "/Users/Viku/Datasets/Medical/Knee"  # Update this with your dataset path
    output_dir = "output_v3"
    
    # Model parameters - optimized for classification
    num_classes = 5
    input_size = (224, 224)  # Keep this if it works well with your data
    batch_size = 4  # Increased since we're not doing segmentation
    num_workers = 2
    prefetch_factor = 2

    in_channels = 1  # Changed to 1 for grayscale
    
    # Class distribution
    class_distribution = {0: 2286, 1: 1046, 2: 1516, 3: 757, 4: 173}
    
    # Training parameters - adjusted for classification
    num_epochs = 100
    learning_rate = 1e-4  # Slightly lower for more stable training
    base_learning_rate = 3e-4
    weight_decay = 2e-4  # Increased for better regularization
    dropout_rate = 0.5  # Increased dropout
    stochastic_depth_prob = 0.3
    validate_every = 2  # Add this to validate every 2 epochs
    attention_log_freq = 500  # Add this to reduce logging frequency
    
    # Class balancing
    class_weights = None  # Will be calculated dynamically

    # Learning rate scheduler
    min_lr = 1e-6
    warmup_epochs = 5  # Reduced since we don't need as long warmup for classification

    # Gradient accumulation to simulate larger batch size
    gradient_accumulation_steps = 2  # Simulates batch size of 32
    
    # Focal Loss parameters - kept as they work well for imbalanced classification
    focal_alpha = 0.75
    focal_gamma = 1.5

    # Mixup parameters
    mixup_alpha = 0.2
    cutmix_alpha = 0.2
    
    # K-fold parameters
    n_folds = 5
    
    # Mixed precision training
    use_amp = False
    
    # Early stopping - adjusted for classification
    patience = 10  

    # Add verbose configuration
    verbose = True  # Enable verbose output by default
        
    # Add gradient logging parameters
    log_gradients = True
    gradient_log_freq = 100
    
    # Device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    @classmethod
    def get_class_weights(cls):
        """Calculate balanced class weights using square root scaling"""
        class_counts = cls.class_distribution
        
        weights = {}
        total_samples = sum(class_counts.values())
        
        for cls_idx, count in class_counts.items():
            # Use square root to moderate the weights
            weights[cls_idx] = math.sqrt(total_samples / (count * len(class_counts)))
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {cls_idx: (weight/weight_sum) * len(weights) 
                for cls_idx, weight in weights.items()}
        
        return weights

    @property
    def class_weights(self):
        """Property to get class weights when needed"""
        return self.get_class_weights()

    def get_cosine_schedule_with_warmup(self, optimizer, num_training_steps):
        """
        Create a schedule with linear warmup and cosine decay.
        
        Args:
            optimizer: The optimizer to use
            num_training_steps (int): Total number of training steps
        
        Returns:
            LambdaLR: The learning rate scheduler
        """
        num_warmup_steps = self.warmup_epochs * (num_training_steps // self.num_epochs)
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(self.min_lr / self.learning_rate, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)

    def create_scheduler(self, optimizer, train_loader):
        """
        Create the learning rate scheduler with warmup.
        
        Args:
            optimizer: The optimizer to use
            train_loader: The training data loader
        
        Returns:
            LambdaLR: The learning rate scheduler
        """
        num_training_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        return self.get_cosine_schedule_with_warmup(optimizer, num_training_steps)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def get_transforms(phase, mean=0.485, std=0.229):
    if phase == 'train':
        return A.Compose([
            A.Resize(Config.input_size[0], Config.input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),  # Reduced from 0.5
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.1,     # Reduced from 0.15
                rotate_limit=15,      # Reduced from 20
                p=0.5                 # Reduced from 0.7
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,  # Reduced from 0.4
                    contrast_limit=0.2,    # Reduced from 0.4
                    p=0.7
                ),
                A.CLAHE(clip_limit=2, p=0.3),  # Reduced clip_limit
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),  # Reduced noise
            ], p=0.5),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2(),
        ])


class KneeXrayDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # First collect all samples
        for grade in range(5):
            grade_path = os.path.join(self.root_dir, str(grade))
            if os.path.exists(grade_path):
                for img_name in os.listdir(grade_path):
                    if img_name.endswith('.png'):
                        self.samples.append((
                            os.path.join(grade_path, img_name),
                            grade
                        ))
                        self.targets.append(grade)
        
        # Calculate mean and std of the dataset
        if phase == 'train':
            self.mean, self.std = self._calculate_stats()
        else:
            # For validation/test phases, use training statistics
            self.mean, self.std = None, None  # Will be set after calculating training stats
    
    def _calculate_stats(self):
        """Calculate mean and std of the dataset specifically for grayscale images"""
        print("Calculating dataset statistics...")
        
        # Use a subset of images for faster calculation
        num_samples = min(100, len(self.samples))
        total_pixels = 0
        sum_pixels = 0
        sum_squared_pixels = 0
        
        for img_path, _ in tqdm(self.samples[:num_samples]):
            # Load image (already in grayscale)
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Update statistics using Welford's online algorithm
            num_pixels = img_array.size
            total_pixels += num_pixels
            sum_pixels += np.sum(img_array)
            sum_squared_pixels += np.sum(img_array ** 2)
        
        # Calculate mean and std
        mean = sum_pixels / total_pixels
        variance = (sum_squared_pixels / total_pixels) - (mean ** 2)
        std = np.sqrt(variance)
        
        print(f"Dataset statistics - Mean: {mean:.3f}, Std: {std:.3f}")
        return mean, std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, grade = self.samples[idx]
        image = Image.open(img_path)  # Already in grayscale
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert single channel to three channels for ResNet
        image = image.repeat(3, 1, 1)
        
        return image, grade

class EnhancedFocalLoss(nn.Module):
    def __init__(self, class_distribution, alpha=0.85, gamma=2.0, label_smoothing=0.1, 
                 reduction='mean', adaptive_gamma=False, normalize_weights=True, 
                 confidence_threshold=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.adaptive_gamma = adaptive_gamma
        self.normalize_weights = normalize_weights
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize weights based on class distribution
        total_samples = sum(class_distribution.values())
        self.initial_weights = {
            cls: (total_samples / (len(class_distribution) * count))
            for cls, count in class_distribution.items()
        }
        
        # Normalize initial weights if requested
        if normalize_weights:
            weight_sum = sum(self.initial_weights.values())
            self.initial_weights = {k: v/weight_sum * len(class_distribution) 
                                  for k, v in self.initial_weights.items()}
        
        self.class_weights = self.initial_weights.copy()
        self.weight_history = {cls: [] for cls in class_distribution.keys()}
        self.accuracy_history = {cls: [] for cls in class_distribution.keys()}
        
        # Parameters for dynamic weight adjustment
        self.current_epoch = 0
        self.warmup_epochs = 5
        self.weight_adjustment_factor = 0.1
        self.max_weight = 10.0
        self.min_weight = 0.1
        self.momentum = 0.9
        
        # Track consecutive poor performance
        self.poor_performance_counter = {cls: 0 for cls in class_distribution.keys()}
        self.performance_threshold = 0.3
        
        print("Initial class weights:", self.initial_weights)

    def update_weights_based_on_performance(self, class_accuracies):
        """
        Dynamically adjust weights based on class performance with memory
        and more aggressive adjustment for consistently poor performers
        """
        if self.current_epoch < self.warmup_epochs:
            return
            
        # Update accuracy history
        for cls, acc in class_accuracies.items():
            self.accuracy_history[cls].append(acc)
        
        # Calculate moving averages
        moving_averages = {}
        for cls in class_accuracies.keys():
            recent_accuracies = self.accuracy_history[cls][-5:]  # Last 5 epochs
            moving_averages[cls] = sum(recent_accuracies) / len(recent_accuracies)
        
        avg_accuracy = sum(moving_averages.values()) / len(moving_averages)
        new_weights = self.class_weights.copy()
        
        for cls, accuracy in moving_averages.items():
            # Update poor performance counter
            if accuracy < self.performance_threshold:
                self.poor_performance_counter[cls] += 1
            else:
                self.poor_performance_counter[cls] = max(0, self.poor_performance_counter[cls] - 1)
            
            # Calculate base adjustment factor
            if accuracy < avg_accuracy * 0.7:  # Underperforming
                # More aggressive adjustment for consistently poor performers
                adjustment_factor = 1.5 * (1 + (avg_accuracy - accuracy))
                if self.poor_performance_counter[cls] > 2:
                    adjustment_factor *= (1 + self.poor_performance_counter[cls] * 0.1)
                
                new_weights[cls] = min(
                    self.class_weights[cls] * adjustment_factor,
                    self.max_weight
                )
            elif accuracy > avg_accuracy * 1.3:  # Overperforming
                decay_factor = max(0.8, 1 - (accuracy - avg_accuracy))
                new_weights[cls] = max(
                    self.class_weights[cls] * decay_factor,
                    self.min_weight
                )
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        normalized_weights = {k: (v/total_weight) * len(new_weights) 
                            for k, v in new_weights.items()}
        
        # Apply momentum for smooth updates
        self.class_weights = {
            k: self.momentum * self.class_weights[k] + (1 - self.momentum) * normalized_weights[k]
            for k in self.class_weights.keys()
        }
        
        # Store weight history
        for cls, weight in self.class_weights.items():
            self.weight_history[cls].append(weight)
        
        # Log significant weight changes
        self._log_weight_changes(moving_averages)
    
    def _log_weight_changes(self, current_accuracies):
        """Log significant weight changes and their reasons"""
        print(f"\nEpoch {self.current_epoch} - Weight Updates:")
        print("-" * 50)
        for cls in self.class_weights.keys():
            initial = self.initial_weights[cls]
            current = self.class_weights[cls]
            accuracy = current_accuracies[cls]
            change = (current - initial) / initial * 100
            
            print(f"Class {cls}:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Weight: {current:.3f} ({change:+.1f}% from initial)")
            print(f"  Poor performance count: {self.poor_performance_counter[cls]}")
            
            if abs(change) > 20:  # Log significant changes
                reason = "increased" if change > 0 else "decreased"
                print(f"  Significant weight {reason} due to "
                      f"{'poor' if change > 0 else 'good'} performance")
    
    def forward(self, inputs, targets):
        """Enhanced focal loss calculation with dynamic weighting"""
        num_classes = inputs.size(1)
        
        # Apply softmax with improved numerical stability
        probs = F.softmax(inputs, dim=1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        
        # One-hot encode targets with label smoothing
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                         self.label_smoothing / num_classes
        
        # Calculate focal loss components
        pt = (targets_one_hot * probs).sum(1)
        
        # Adaptive gamma if enabled
        if self.adaptive_gamma:
            confidence_mask = pt > self.confidence_threshold
            gamma = torch.ones_like(pt) * self.gamma
            gamma[confidence_mask] = self.gamma * 0.5  # Reduce focusing for high-confidence predictions
            focal_weight = (1 - pt).pow(gamma)
        else:
            focal_weight = (1 - pt).pow(self.gamma)
        
        # Apply current class weights
        weights = torch.tensor([self.class_weights[i] for i in range(num_classes)],
                             device=self.device)
        class_weights = weights[targets]
        focal_weight = focal_weight * class_weights
        
        loss = -self.alpha * focal_weight * torch.log(pt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    
    def update_epoch(self, epoch):
        """Update the current epoch number"""
        self.current_epoch = epoch
        self.weight_adjustment_factor = 0.1 * (1 + epoch / 50)  # Gradually increase
        
    def get_weight_changes(self):
        """Get a summary of weight changes over time"""
        changes = {}
        for cls in self.class_weights.keys():
            initial = self.initial_weights[cls]
            current = self.class_weights[cls]
            changes[cls] = {
                'initial': initial,
                'current': current,
                'change_percent': (current - initial) / initial * 100
            }
        return changes
        
class AttentionVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / 'attention_maps'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self, model, data_loader, device, epoch, num_samples=5):
        """Visualizes attention maps for classification."""
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if i >= num_samples:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # Get model outputs with attention maps
                outputs = model(images, return_attention=True)
                
                # Handle different return formats
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                    attention_maps = outputs[-1]
                else:
                    predictions = outputs
                    attention_maps = None
                
                # Plot results
                if attention_maps is not None:
                    for j, (img, attn, pred, label) in enumerate(zip(images, attention_maps, predictions.argmax(1), labels)):
                        self._plot_attention(img, attn, pred.item(), label.item(), epoch, f"{i}_{j}")

    def _plot_attention(self, image, attention_map, prediction, label, epoch, idx):
        """Plots a single attention map."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original image
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        ax1.imshow(img_np)
        ax1.set_title(f'Pred: {prediction}, True: {label}')
        ax1.axis('off')
        
        # Plot attention map
        if len(attention_map.shape) > 2:
            attention_map = attention_map.mean(0)  # Average over heads if multi-head
        sns.heatmap(attention_map.cpu(), ax=ax2, cmap='viridis')
        ax2.set_title('Attention Map')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'attention_epoch_{epoch}_sample_{idx}.png')
        plt.close()

def check_gradients(model, grad_norm):
    has_nan = False
    has_inf = False
    max_grad = 0
    min_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                has_inf = True
                print(f"Inf gradient in {name}")
            max_grad = max(max_grad, param.grad.max().item())
            min_grad = min(min_grad, param.grad.min().item())
    
    return {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'max_grad': max_grad,
        'min_grad': min_grad,
        'grad_norm': grad_norm
    }

def validate_loss(loss, outputs, targets):
    if not torch.isfinite(loss):
        return {
            'is_valid': False,
            'reason': f"Loss is {loss.item()}",
            'details': {
                'outputs_min': outputs.min().item(),
                'outputs_max': outputs.max().item(),
                'target_dist': torch.bincount(targets).tolist()
            }
        }
    return {'is_valid': True}

def log_lr_changes(scheduler, epoch, batch_idx, total_batches):
    lr = scheduler.get_last_lr()[0]
    if batch_idx == 0 or batch_idx == total_batches - 1:
        print(f"\nEpoch {epoch + 1}: LR = {lr:.6f}")

def analyze_model_weights(model, epoch):
    if epoch % 5 == 0:  # Every 5 epochs
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"\n{name}:")
                print(f"Mean: {param.data.mean().item():.4f}")
                print(f"Std: {param.data.std().item():.4f}")
                print(f"Max: {param.data.max().item():.4f}")
                print(f"Min: {param.data.min().item():.4f}")

def verify_data_distribution(dataset, phase='train'):
    """Verify class distribution in dataset"""
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1
    
    print(f"\n{phase.capitalize()} Dataset Class Distribution:")
    print("-" * 40)
    for class_idx in range(Config.num_classes):
        count = class_counts[class_idx]
        percentage = (count / len(dataset)) * 100
        print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")
    print("-" * 40)
    return class_counts

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Set the starting learning rate
        self.lr_start = 1e-7
        self.lr_end = 1e-2  # Reduced from 10 to 1 for better stability
        self.num_iter = 50  # Reduced from 100 to 30 for faster execution
        
        # Store the best loss for comparison
        self.best_loss = None
        
    def range_test(self, train_loader):
        # Calculate the multiplier for learning rate increase
        mult = (self.lr_end / self.lr_start) ** (1/self.num_iter)
        lr = self.lr_start
        
        # Save original learning rate
        orig_lr = self.optimizer.param_groups[0]['lr']
        
        # Storage for learning rates and corresponding losses
        lrs = []
        losses = []
        best_loss = float('inf')
        
        # Early stopping parameters
        patience = 10
        min_loss_improvement = 0.25
        consecutive_bad_loss = 0
        
        iterator = tqdm(range(self.num_iter))
        
        try:
            for iteration in iterator:
                # Set learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                # Train on one batch
                batch_loss = self._train_one_batch(train_loader)
                
                # Store the values
                lrs.append(lr)
                losses.append(batch_loss)
                
                # Update the learning rate
                lr *= mult
                
                # Early stopping logic
                if batch_loss < best_loss * (1 - min_loss_improvement):
                    best_loss = batch_loss
                    consecutive_bad_loss = 0
                else:
                    consecutive_bad_loss += 1
                
                # Stop if the loss is exploding or not improving
                if (batch_loss > 4 * best_loss or 
                    torch.isnan(torch.tensor(batch_loss)) or 
                    consecutive_bad_loss >= patience):
                    break
                    
                # Update progress bar
                iterator.set_description(f'Loss: {batch_loss:.4f}, LR: {lr:.7f}')
                
                # Clear memory every few iterations
                if iteration % 5 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error during range test: {str(e)}")
            
        finally:
            # Reset learning rate to original value
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = orig_lr
            
            # Clear any remaining memory
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return lrs, losses
    
    def _train_one_batch(self, train_loader):
        self.model.train()
        batch = next(iter(train_loader))
        images, targets = batch
        images, targets = images.to(self.device), targets.to(self.device)
        
        try:
            # Forward pass with error handling
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in training batch: {str(e)}")
            return float('inf')  # Return high loss to trigger early stopping
        
        finally:
            # Clear memory
            del images, targets, outputs
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
    def plot(self, lrs, losses):
        # Convert to numpy arrays and handle potential empty lists
        if not lrs or not losses:
            print("No valid learning rates to plot")
            return 3e-4  # Return default learning rate
            
        lrs = np.array(lrs)
        losses = np.array(losses)
        
        try:
            # Remove infinite or NaN values
            mask = np.isfinite(losses)
            lrs = lrs[mask]
            losses = losses[mask]
            
            if len(lrs) < 2:
                print("Insufficient valid data points for analysis")
                return 3e-4
            
            # Simple moving average for smoothing
            window_size = min(5, len(losses) // 3)
            if window_size < 2:
                window_size = 2
                
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_lrs = lrs[window_size-1:]
            
            # Find the learning rate with minimum loss
            min_loss_idx = np.argmin(smoothed_losses)
            suggested_lr = smoothed_lrs[min_loss_idx]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.semilogx(smoothed_lrs, smoothed_losses)
            plt.plot(suggested_lr, smoothed_losses[min_loss_idx], 'ro', 
                    label=f'Minimum loss\nLR={suggested_lr:.2e}')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True)
            plt.legend()
            
            # Save plot instead of showing it
            plt.savefig('lr_finder_plot.png')
            plt.close()
            
            return suggested_lr
            
        except Exception as e:
            print(f"Error during plotting: {str(e)}")
            return 3e-4
        
        finally:
            plt.close('all')  # Ensure all plots are closed

def find_optimal_lr(model, train_loader, device, start_lr=1e-7, end_lr=10, num_iter=100, config=None):
    """Helper function to find optimal learning rate"""
    try:
        # Calculate class distribution from the train_loader
        class_distribution = defaultdict(int)
        for _, labels in train_loader:
            for label in labels:
                class_distribution[label.item()] += 1

        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        # Create criterion with class distribution directly
        criterion = EnhancedFocalLoss(
            class_distribution=class_distribution,
            alpha=0.75 if config is None else config.focal_alpha,
            gamma=2.5 if config is None else config.focal_gamma,
            label_smoothing=0.1
        ).to(device)
        
        # Initialize LR finder
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.lr_start = start_lr
        lr_finder.lr_end = end_lr
        lr_finder.num_iter = num_iter
        
        # Run range test
        lrs, losses = lr_finder.range_test(train_loader)
        
        # Plot and get suggestion
        suggested_lr = lr_finder.plot(lrs, losses)
        
        # Clean up
        del lr_finder
        torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()
        
        return suggested_lr
        
    except Exception as e:
        print(f"\nError in learning rate finder: {str(e)}")
        return 3e-4  # Default fallback
    
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()

# Helper function to update dataloader batch size
def update_dataloader(loader, new_batch_size):
    return DataLoader(
        loader.dataset,
        batch_size=new_batch_size,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        prefetch_factor=loader.prefetch_factor,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers
    )

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-7):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                   for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay 
                   for base_lr in self.base_lrs]


def clip_and_log_gradients(model, clip_value, norm_type, config):
    """Helper function to handle gradient clipping and logging"""
    try:
        # Two-stage gradient clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_value,
            norm_type=norm_type
        )
        
        if getattr(config, 'verbose', False) and grad_norm > clip_value * 0.9:
            print(f"Gradient norm {grad_norm:.2f} clipped to {clip_value}")
            
        if getattr(config, 'log_gradients', False):
            check_gradients(model, grad_norm)
            
        return grad_norm
        
    except RuntimeError as e:
        print(f"Error during gradient clipping: {str(e)}")
        return 0.0
    
def initialize_training(model, config):
    """
    Initialize model weights, optimizer, and training components optimized for 16GB M1 Mac.
    """
    # Initialize model weights with smaller variance
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Update config with M1-optimized parameters
    config.initial_batch_size = 4  # Smaller batch size for MPS
    config.grad_accumulation_steps = 16  # Increased for effective batch size of 64
    config.grad_clip_value = 5.0  # Increased but still conservative
    config.grad_clip_norm_type = 2.0
    
    # Initialize optimizer with conservative learning rate and increased stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6,  # Very conservative initial learning rate
        betas=(0.9, 0.999),  # Default betas work well
        eps=1e-8,
        weight_decay=0.02,  # Increased weight decay for stability
        amsgrad=True  # Enable AMSGrad for better convergence
    )
    
    # Scheduler with longer warmup
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=20,  # Extended warmup period
        max_epochs=config.num_epochs,
        warmup_start_lr=1e-7,
        eta_min=1e-7
    )
    
    # Mixed precision not needed for MPS
    scaler = torch.amp.GradScaler(enabled=False)
    
    # Initialize criterion with adjusted parameters
    criterion = EnhancedFocalLoss(
        class_distribution=config.class_distribution,
        alpha=0.75,  # Reduced from 0.85 for more stable training
        gamma=1.5,   # Reduced from 2.0 for smoother convergence
        label_smoothing=0.15,  # Increased for better regularization
        reduction='mean',
        adaptive_gamma=True,
        normalize_weights=True,
        confidence_threshold=0.6  # Lowered threshold for early training
    ).to(config.device)
    
    return optimizer, scheduler, scaler, criterion

def training_step(model, images, targets, optimizer, criterion, scaler, config, batch_idx, epoch):
    """
    Memory-efficient training step for M1 Mac.
    """
    try:
        # Scale loss by accumulation steps
        loss_scale = 1.0 / config.grad_accumulation_steps
        
        # Update criterion's epoch
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        
        # Forward pass
        outputs, attention_maps = model(images, return_attention=True)
        loss = criterion(outputs, targets) * loss_scale
        
        # Backward pass without AMP (not needed for MPS)
        loss.backward()
        
        # Gradient accumulation with monitoring
        if (batch_idx + 1) % config.grad_accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_value,
                norm_type=config.grad_clip_norm_type
            )
            
            if grad_norm > config.grad_clip_value:
                print(f"Gradient norm {grad_norm:.2f} clipped to {config.grad_clip_value}")
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Clear MPS cache periodically
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        return loss, outputs, attention_maps
        
    except RuntimeError as e:
        print(f"Error in training step: {str(e)}")
        return None, None, None

def train_model(config):
    print("\nInitializing Training...")

    # Enable mixed precision training based on available device
    use_amp = hasattr(torch.amp, 'autocast') and not torch.backends.mps.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    # Create full dataset with proper statistics
    full_dataset = KneeXrayDataset(
        root_dir=config.data_root,
        phase='train',
        transform=None  # Don't apply transforms yet
    )

    # Get dataset statistics
    mean, std = full_dataset.mean, full_dataset.std

    # Now create the actual training dataset with proper transforms
    full_dataset = KneeXrayDataset(
        root_dir=config.data_root,
        phase='train',
        transform=get_transforms('train', mean=mean, std=std)
    )
    
    # Verify data distribution before splitting
    class_counts = verify_data_distribution(full_dataset, 'full')
    
    # Update class distribution in config
    config.class_distribution = dict(class_counts)
    
    # Initialize k-fold cross validation with stratification
    kf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    results = []
    # Dynamic batch sizing with memory management
    initial_batch_size = 16
    max_batch_size = config.batch_size
    current_batch_size = initial_batch_size
    grad_accumulation_steps = 4  # Initial gradient accumulation steps
    
    # Get all targets for stratification
    all_targets = [target for _, target in full_dataset]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)), all_targets)):
        print(f'\nTraining Fold {fold + 1}/{config.n_folds}')
        print('-' * 50)

        # Initialize monitoring
        monitor = TrainingMonitor(Path(config.output_dir) / f'fold_{fold}')
        
        # Create train/val datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Verify split distribution
        print("\nVerifying data split distribution...")
        train_counts = verify_data_distribution(train_dataset, 'train')
        val_counts = verify_data_distribution(val_dataset, 'validation')
        
        # Calculate class weights for weighted sampling
        class_weights = config.get_class_weights()
        print("\nClass weights:", class_weights)
        
        # Create weighted sampler for training
        train_targets = [all_targets[i] for i in train_idx]
        train_weights = [class_weights[t] for t in train_targets]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_idx),
            replacement=True
        )
        
       # Create weighted sampler
        train_targets = [all_targets[i] for i in train_idx]
        train_weights = [class_weights[t] for t in train_targets]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_idx),
            replacement=True
        )
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            prefetch_factor=2,  # Reduced for memory efficiency
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize model with improved architecture
        model = HybridModel(
        num_classes=config.num_classes,
        use_residual=True,
        use_skip_connections=True
        ).to(config.device)
    
        
        optimizer, scheduler, scaler, criterion = initialize_training(model, config)
        
        best_val_loss = float('inf')
        smoothed_val_loss = None
        patience_counter = 0
        best_epoch_metrics = {}

        # Before training, find optimal learning rate
        print( 'Finding Optimal Learning Rate: ')
        # Create a small dataloader for LR finding
        lr_find_loader = DataLoader(
            train_dataset,
            batch_size=8,  # Smaller batch size for LR finding
            sampler=train_sampler,
            num_workers=1,
            pin_memory=True
        )

        # Find optimal learning rate
        suggested_lr = find_optimal_lr(
            model=model,
            train_loader=lr_find_loader,
            device=config.device,
            start_lr=1e-6,  # Increase from 1e-7
            end_lr=1e-3,    # Decrease from 1e-2
            num_iter=100    # Increase from 50 for better resolution
        )

        # Update the config with the found learning rate
        config.learning_rate = suggested_lr

        print(f"Optimal learning rate found: {suggested_lr:.2e}")
        
        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print("=" * 50)

            criterion.update_epoch(epoch)

            # Optionally, print weight change analysis
            if epoch % 5 == 0:  # Every 5 epochs
                print("\nWeight change analysis:")
                print(criterion.get_weight_changes())
            
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_class_correct = {i: 0 for i in range(config.num_classes)}
            train_class_total = {i: 0 for i in range(config.num_classes)}
            batch_grad_norms = []
            
            optimizer.zero_grad()  # Zero gradients at start of epoch

            progress_bar = tqdm(train_loader, desc='Training')
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images = images.to(config.device)
                targets = targets.to(config.device)
                
                # Perform training step
                loss, outputs, attention_maps = training_step(
                    model=model,
                    images=images,
                    targets=targets,
                    optimizer=optimizer,
                    criterion=criterion,
                    scaler=scaler,
                    config=config,
                    batch_idx=batch_idx,
                    epoch=epoch
                )
                
                if loss is not None and outputs is not None:
                    # Calculate accuracy metrics
                    _, predicted = outputs.max(1)
                    correct = predicted.eq(targets)
                    train_total += targets.size(0)
                    train_correct += correct.sum().item()
                    
                    # Update per-class metrics
                    for class_idx in range(config.num_classes):
                        mask = targets == class_idx
                        train_class_total[class_idx] += mask.sum().item()
                        train_class_correct[class_idx] += (correct & mask).sum().item()
                    
                    train_loss += loss.item() * config.grad_accumulation_steps
                    
                    # Log training metrics
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * config.grad_accumulation_steps:.4f}",
                        'acc': f"{100.0 * train_correct / train_total:.2f}%",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                
                # Memory management for MPS
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            scheduler.step()
            
            if epoch % config.validate_every == 0:
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                val_class_correct = {i: 0 for i in range(config.num_classes)}
                val_class_total = {i: 0 for i in range(config.num_classes)}

                all_val_targets = []
                all_val_predictions = []
                all_val_probabilities = []
                all_val_inputs = []
                all_attention_maps = []

                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc='Validation')):
                        images, targets = images.to(config.device), targets.to(config.device)
                        
                        # Forward pass
                        outputs, attention_maps = model(images, return_attention=True)
                        all_val_inputs.append(outputs.cpu()) # Store raw logits
                        probabilities = F.softmax(outputs, dim=1) # Calculate probabilities separately
                        predicted = torch.argmax(probabilities, dim=1)  # Add this line to get predictions

                        # Store for later analysis 
                        all_val_targets.extend(targets.cpu().numpy())
                        all_val_predictions.extend(predicted.cpu().numpy())
                        all_val_probabilities.extend(probabilities.cpu().numpy())
                        all_attention_maps.append(attention_maps.cpu())

                        # Update running metrics
                        val_total += targets.size(0)
                        correct = predicted.eq(targets)
                        val_correct += correct.sum().item()
                        
                        # Update per-class metrics
                        for class_idx in range(config.num_classes):
                            mask = targets == class_idx
                            val_class_total[class_idx] += mask.sum().item()
                            val_class_correct[class_idx] += (correct & mask).sum().item()
                        
                        # Log attention maps periodically
                        if batch_idx % config.attention_log_freq == 0:
                            monitor.plot_attention_maps(
                                attention_maps.detach(),
                                images.detach(),
                                predicted.detach(),
                                targets.detach(),
                                epoch,
                                batch_idx
                            )

                # Convert lists to numpy arrays
                all_val_targets = np.array(all_val_targets)
                all_val_predictions = np.array(all_val_predictions)
                all_val_probabilities = np.vstack(all_val_probabilities)
                all_val_inputs = torch.cat(all_val_inputs, dim=0)

                # Calculate and print metrics
                val_metrics = print_and_calculate_metrics(
                    targets=all_val_targets,
                    predictions=all_val_predictions,
                    probabilities=all_val_probabilities,
                    loss_function=criterion,
                    inputs=all_val_inputs,
                    class_correct=val_class_correct,
                    class_total=val_class_total,
                    phase='validation',
                    epoch=epoch,
                    epoch_time=time.time() - epoch_start_time,
                    config=config
                )

                # Plot confusion matrix and save classification report
                monitor.plot_confusion_matrix(all_val_targets, all_val_predictions, epoch)
                monitor.save_classification_report(all_val_targets, all_val_predictions, epoch)

                # Log metrics using available TrainingMonitor methods
                # Log epoch metrics
                is_best = monitor.log_epoch(
                    epoch=epoch,
                    train_metrics={'loss': train_loss / len(train_loader)},
                    val_metrics=val_metrics,
                    epoch_time=time.time() - epoch_start_time
                )

                # Log batch-level metrics if available
                if batch_grad_norms:
                    for grad_norm in batch_grad_norms:
                        monitor.log_batch(
                            batch_idx=batch_idx,
                            loss=train_loss / len(train_loader),
                            grad_norm=grad_norm,
                            lr=optimizer.param_groups[0]['lr']
                        )

                # Log any scalar metrics
                monitor.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                # Process attention maps
                all_attention_maps = torch.cat(all_attention_maps, dim=0)
                monitor.log_attention_metrics(all_attention_maps, epoch)

                # Check for best model
                is_best = smoothed_val_loss < best_val_loss
                if is_best:
                    print("\n=> Saving new best model")
                    best_val_loss = smoothed_val_loss
                    patience_counter = 0
                    best_epoch_metrics = {
                        'epoch': epoch,
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics.get('accuracy', val_metrics.get('acc', 0.0)),
                        'val_f1': val_metrics.get('f1_weighted', val_metrics.get('f1_score', 0.0))
                    }
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics.get('accuracy', val_metrics.get('acc', 0.0))
                    }, Path(config.output_dir) / f'fold_{fold}_best.pth')
                else:
                    patience_counter += 1

            # Update scheduler
            scheduler.step()

            # Dynamic batch size adjustment (every 10 epochs)
            if epoch > 0 and epoch % 10 == 0 and current_batch_size < max_batch_size:
                try:
                    current_batch_size *= 2
                    grad_accumulation_steps = max(1, grad_accumulation_steps // 2)
                    train_loader = update_dataloader(train_loader, current_batch_size)
                    val_loader = update_dataloader(val_loader, current_batch_size)
                except RuntimeError:  # Memory error
                    current_batch_size //= 2
                    grad_accumulation_steps *= 2

            # Early stopping check
            if patience_counter >= config.patience:
                print(f'\nEarly stopping triggered after {config.patience} epochs without improvement')
                break

            # Cleanup
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Modify the results dictionary to use consistent key names
        results.append({
            'fold': fold,
            'best_val_loss': monitor.best_val_loss,
            'best_val_acc': monitor.best_val_acc,
            'best_epoch': best_epoch_metrics
        })

    # Return results with proper key handling
    final_results = {
        'results': results,
        'avg_val_loss': np.mean([r['best_val_loss'] for r in results]),
        'avg_val_acc': np.mean([r['best_val_acc'] for r in results]),
        'best_fold': min(range(len(results)), key=lambda i: results[i]['best_val_loss'])
    }

    return final_results

def print_and_calculate_metrics(targets, predictions, probabilities, loss_function=None, inputs=None, 
                              class_correct=None, class_total=None, phase='train', epoch=None, 
                              epoch_time=None, config=None):
    """Combined function for calculating and printing metrics."""
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1_score': f1_score(targets, predictions, average='weighted'),
        'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
        'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(targets, predictions)
    }
    
    # Add both keys for compatibility
    metrics['acc'] = metrics['accuracy']
    metrics['f1_weighted'] = metrics['f1_score']
    
    # Calculate class-specific metrics
    n_classes = probabilities.shape[1]
    for i in range(n_classes):
        try:
            class_mask = targets == i
            if np.any(class_mask):
                metrics[f'class_{i}_precision'] = precision_score(
                    targets == i, predictions == i, average='binary', zero_division=0
                )
                metrics[f'class_{i}_recall'] = recall_score(
                    targets == i, predictions == i, average='binary', zero_division=0
                )
                metrics[f'class_{i}_f1'] = f1_score(
                    targets == i, predictions == i, average='binary', zero_division=0
                )
        except Exception as e:
            print(f"Warning: Error calculating metrics for class {i}: {str(e)}")
            metrics[f'class_{i}_precision'] = metrics[f'class_{i}_recall'] = metrics[f'class_{i}_f1'] = 0.0

    # Calculate loss if provided
    if loss_function is not None and inputs is not None:
        try:
            # Ensure inputs are raw logits
            inputs_tensor = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs)
            targets_tensor = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
            
            # Move to correct device
            device = next(loss_function.parameters()).device
            inputs_tensor = inputs_tensor.to(device)
            targets_tensor = targets_tensor.to(device)
            
            # Calculate loss
            with torch.no_grad():
                loss_val = loss_function(inputs_tensor, targets_tensor)
                metrics['loss'] = loss_val.item()
                
        except Exception as e:
            print(f"Warning: Could not calculate loss: {str(e)}")
            print(f"Input shape: {inputs.shape if hasattr(inputs, 'shape') else 'unknown'}")
            print(f"Target shape: {targets.shape if hasattr(targets, 'shape') else 'unknown'}")
            raise  # Re-raise the exception to see the full error
        
    # Print results if requested
    if epoch is not None and config is not None:
        print(f"\n{'='*80}\n{phase.capitalize()} Metrics - Epoch {epoch+1}")
        if epoch_time:
            print(f"Epoch duration: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
        
        print(f"\nOverall Metrics:")
        print(f"Loss: {metrics.get('loss', 0):.4f}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        if class_correct and class_total:
            print(f"\nClass-wise Metrics:")
            print("-" * 60)
            print(f"{'Class':^10} | {'Samples':^10} | {'Correct':^10} | {'Accuracy':^12}")
            print("-" * 60)
            for class_idx in range(config.num_classes):
                total = class_total[class_idx]
                correct = class_correct[class_idx]
                accuracy = (correct / total * 100) if total > 0 else 0
                print(f"{class_idx:^10} | {total:^10} | {correct:^10} | {accuracy:^11.2f}%")
    
    return metrics

def main():
    # Initialize configuration
    config = Config()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    print("Starting training process...")
    try:
        results = train_model(config)
        
        # Save final results
        final_results = {
            'avg_val_loss': float(results['avg_val_loss']),
            'avg_val_acc': float(results['avg_val_acc']),
            'best_fold': int(results['best_fold']),
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_val_loss': float(r['best_val_loss']),
                    'best_val_acc': float(r['best_val_acc'])
                }
                for r in results['results']
            ]
        }
        
        # Save results to JSON
        with open(output_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=4)
            
        print("\nTraining completed successfully!")
        print(f"Results saved to {output_dir / 'final_results.json'}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()