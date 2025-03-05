import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import time
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
import gc
from torch.optim.lr_scheduler import OneCycleLR
import sys
import copy
from pathlib import Path
import augmentation as A
from albumentations.pytorch import ToTensorV2



def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

from dataset import KneeDataset
from model_arch import HybridModel
from trainingMonitor import TrainingMonitor
from config import config
from augmentation import AugmentationPipeline, MixupCutmixCollator


os.environ['TQDM_DISABLE'] = '0'  # Set to '1' to disable tqdm completely

class_counts = [2286, 1046, 1516, 757, 173]


# Add ExponentialMovingAverage for model weights
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1. - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]  


class ClassificationOnlyLoss(nn.Module):
    def __init__(self, class_counts, device='cuda', gamma=2.0, aux_weight=0.3):
        super().__init__()
        self.device = device
        self.aux_weight = aux_weight
        
        # Convert counts to tensor
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Calculate median for reference
        counts_list = [class_counts[i] for i in range(num_classes)]
        counts_tensor = torch.tensor(counts_list, dtype=torch.float32)
        median_count = torch.median(counts_tensor)
        
        # Initialize alpha (class weights) with inverse frequency
        alpha = torch.zeros(num_classes, dtype=torch.float32)
        for i in range(num_classes):
            count = class_counts[i]
            # Base inverse frequency weighting
            weight = total_samples / (count * num_classes)
            
            # Additional weighting for underrepresented classes
            if count < median_count * 0.1:  # Severe underrepresentation (class 4)
                weight *= 2.5
            elif count < median_count * 0.4:  # Moderate underrepresentation (classes 1 and 3)
                weight *= 1.8
                
            alpha[i] = weight
            
        # Normalize weights
        self.alpha = (alpha / alpha.sum() * num_classes).to(device)
        self.gamma = gamma
        
        # Store rare classes for dynamic scaling
        self.rare_classes = torch.tensor([i for i, c in class_counts.items() 
                                        if c < median_count * 0.4]).to(device)
        
        # Regular cross entropy for auxiliary heads
        self.aux_criterion = nn.CrossEntropyLoss(weight=self.alpha)
    
    def forward(self, model_outputs, targets):
        """
        Handle all outputs from the HybridModel while focusing only on classification
        
        Args:
            model_outputs: Tuple of (seg_output, cls_output, aux_seg_outputs, aux_cls_outputs)
            targets: Classification targets
        """
        # Unpack outputs based on training/inference mode
        if isinstance(model_outputs, tuple) and len(model_outputs) >= 4:
            # Training mode with all outputs
            _, cls_out, _, aux_cls_outputs, _ = model_outputs
        else:
            # Inference mode with only main outputs
            _, cls_out, _ = model_outputs
            aux_cls_outputs = None
        
        # Get probabilities for main classification
        logit = F.log_softmax(cls_out, dim=1)
        pt = torch.exp(logit)
        
        # Get target probabilities
        target_pt = pt.gather(1, targets.view(-1, 1))
        
        # Compute focal loss with alpha balancing
        alpha = self.alpha.gather(0, targets)
        focal_loss = -alpha * torch.pow(1 - target_pt, self.gamma) * logit.gather(1, targets.view(-1, 1))
        
        # Additional weighting for rare classes
        rare_mask = torch.isin(targets, self.rare_classes)
        if rare_mask.any():
            focal_loss[rare_mask] *= 1.5
            
            # Add auxiliary term for rare classes
            aux_rare_loss = -torch.log(target_pt[rare_mask]).mean()
            focal_loss[rare_mask] += 0.3 * aux_rare_loss
        
        main_loss = focal_loss.mean()
        
        # Handle auxiliary classification outputs if present
        if aux_cls_outputs is not None and len(aux_cls_outputs) > 0:
            aux_loss = 0.0
            for aux_out in aux_cls_outputs:
                aux_loss += self.aux_criterion(aux_out, targets)
            aux_loss /= len(aux_cls_outputs)
            return main_loss + self.aux_weight * aux_loss
            
        return main_loss

class ClassificationAugmentor:
    """
    Augmentation pipeline optimized for classification tasks.
    """
    def __init__(self, config):
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.input_size,
                width=config.input_size,
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(config.input_size, config.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
class ClassBalancedSampler:
    def __init__(self, labels, num_samples=None, beta=0.9999):
        """
        Args:
            labels: List of class labels
            num_samples: Number of samples to draw. If None, uses len(labels)
            beta: Smoothing parameter for effective number of samples
        """
        self.labels = np.array(labels)
        self.num_samples = len(labels) if num_samples is None else min(num_samples, len(labels))
        self.beta = beta
        
        # Calculate per-class sample count
        self.class_counts = np.bincount(labels)
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(self.beta, self.class_counts)
        effective_num = np.where(effective_num == 0, 1.0, effective_num)  # Avoid division by zero
        
        # Calculate weights per class
        self.class_weights = (1.0 - self.beta) / effective_num
        
        # Normalize weights
        self.class_weights = self.class_weights / np.sum(self.class_weights) * len(self.class_weights)
        
        # Create sample weights
        self.sample_weights = self.class_weights[self.labels]
        
        # Normalize sample weights
        self.sample_weights = self.sample_weights / self.sample_weights.sum()

    def __iter__(self):
        # Generate random indices with replacement using normalized weights
        rand_indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.sample_weights
        )
        return iter(rand_indices.tolist())

    def __len__(self):
        return self.num_samples

def create_balanced_sampler(dataset, indices):
    labels = [dataset.samples[i]['grade'] for i in indices]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts[labels]
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


def find_lr_for_grayscale(model, train_loader, criterion, optimizer, device, 
                         num_iter=100, start_lr=1e-7, end_lr=1e-1):
    """
    Wrapper function for finding learning rate specifically for grayscale images
    """
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    try:
        print("Starting learning rate search...")
        # Ensure model is in training mode
        model.train()
        
        lrs, losses = lr_finder.range_test(
            train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter
        )
        
        if len(lrs) < 20:  # If search ended too early
            print("Learning rate search ended early. Trying more conservative range...")
            lr_finder.reset()
            lrs, losses = lr_finder.range_test(
                train_loader,
                start_lr=1e-8,  # Lower start
                end_lr=1e-2,    # Lower end
                num_iter=num_iter
            )
        
        # Plot the results
        try:
            lr_finder.plot(skip_start=10, skip_end=5)
        except Exception as e:
            print(f"Warning: Could not plot learning rate results: {str(e)}")
        
        suggested_lr = lr_finder.suggest(skip_start=10, skip_end=5)
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        
        return suggested_lr
        
    except Exception as e:
        print(f"Error during learning rate finding: {str(e)}")
        # Return a default learning rate if the finder fails
        return 1e-4
        
    finally:
        lr_finder.reset()
        
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.reset()
        
    def reset(self):
        """Reset the learning rate finder state"""
        self.lrs = []  # Store learning rates
        self.losses = []  # Store corresponding losses
        self.best_loss = float('inf')
        # Save model/optimizer state
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
    def restore(self):
        """Restore model/optimizer state"""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """
        Performs the learning rate range test
        """
        # Reset model/optimizer to initial state
        self.restore()
        
        current_lr = start_lr
        mult = (end_lr / start_lr) ** (1/num_iter)
        
        # Initialize running average for loss smoothing
        avg_loss = 0
        beta = 0.98  # Smoothing factor
        
        # Initialize progress bar
        pbar = tqdm(total=num_iter, desc="Finding optimal learning rate", unit='step')
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= num_iter:
                    break
                    
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                    
                # Training step
                self.optimizer.zero_grad()
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                grades = batch['grade'].to(self.device)
                
                # Forward pass
                if self.model.training:
                    # During training, the model returns 5 values
                    seg_out, cls_out, aux_seg_outputs, aux_cls_outputs, _ = self.model(images)
                else:
                    # During inference, the model returns 3 values
                    seg_out, cls_out, _ = self.model(images)
                
                # Calculate loss using main outputs only
                loss = self.criterion(seg_out, cls_out, masks, grades)
                
                # Backward pass
                if not loss.requires_grad:
                    raise RuntimeError("Loss does not require gradients. Check if model parameters require gradients.")
                    
                loss.backward()
                self.optimizer.step()
                
                # Update tracking
                loss_value = loss.item()
                avg_loss = beta * avg_loss + (1-beta) * loss_value
                smoothed_loss = avg_loss / (1 - beta**(batch_idx + 1))
                
                self.lrs.append(current_lr)
                self.losses.append(smoothed_loss)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'lr': f"{current_lr:.2e}", 'loss': f"{smoothed_loss:.4f}"})
                
                current_lr *= mult
                
        finally:
            pbar.close()
            self.restore()
            
        return self.lrs, self.losses
    
    def plot(self, skip_start=10, skip_end=5, save_path='lr_finder_plot.png'):
        """
        Plot the learning rate finder results with improved error handling
        
        Args:
            skip_start: number of batches to skip at the start
            skip_end: number of batches to skip at the end
            save_path: path to save the plot
        """
        if len(self.lrs) == 0 or len(self.losses) == 0:
            raise ValueError("No learning rate test data available. Run range_test() first.")
            
        if len(self.lrs) <= (skip_start + skip_end):
            raise ValueError(f"Not enough data points to plot. Have {len(self.lrs)} points, but skip_start={skip_start} and skip_end={skip_end}")
            
        # Trim the data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        if len(losses) == 0:
            raise ValueError("No valid loss values to plot after trimming")
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Highlight minimum loss only if we have valid data
        min_loss_idx = np.argmin(losses)
        plt.plot(lrs[min_loss_idx], losses[min_loss_idx], 'ro', label='Min Loss')
        
        # Find and highlight suggested learning rate
        try:
            suggested_lr = self.suggest(skip_start=skip_start, skip_end=skip_end)
            suggested_idx = lrs.index(suggested_lr)
            plt.plot(suggested_lr, losses[suggested_idx], 'go', label='Suggested LR')
            plt.legend()
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not plot suggested learning rate: {str(e)}")
        
        plt.savefig(save_path)
        plt.close()
        
    def suggest(self, skip_start=10, skip_end=5):
        """
        Suggest the optimal learning rate based on the steepest descent
        
        Args:
            skip_start: number of batches to skip at the start
            skip_end: number of batches to skip at the end
        """
        if len(self.losses) == 0:
            raise ValueError("No learning rate test has been conducted")
            
        # Trim the data
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        
        # Smooth the loss curve
        window_size = 5
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate gradients
        gradients = np.gradient(smoothed_losses)
        
        # Find the point of steepest descent
        min_gradient_idx = np.argmin(gradients)
        
        # More conservative learning rate selection
        suggested_lr = lrs[min_gradient_idx] * 0.1  # Use 1/10th of the steepest point
        
        return suggested_lr

class AttentionVisualizer:
    """Utility to visualize and save attention maps."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.attention_dir = os.path.join(output_dir, 'attention_maps')
        os.makedirs(self.attention_dir, exist_ok=True)

    def denormalize_image(self, image):
        image = image.clone() * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        return torch.clamp(image, 0, 1)

    def plot_attention_map(self, image, attention_map, prediction, true_grade, epoch, idx):
        plt.switch_backend('agg')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        display_image = self.denormalize_image(image)
        axes[0].imshow(display_image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title(f'Prediction: {prediction}, True: {true_grade}')
        axes[0].axis('off')

        attention_display = attention_map.mean(0).detach().cpu().numpy()
        sns.heatmap(attention_display, cmap='viridis', ax=axes[1])
        axes[1].set_title('Attention Map')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.attention_dir, f'epoch_{epoch}_sample_{idx}.png')
        plt.savefig(save_path)
        plt.close(fig)

def plot_metrics(history, output_dir):
    plt.switch_backend('agg')
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

# Gradient accumulation wrapper
class GradientAccumulator:
    def __init__(self, model, optimizer, steps=4):
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.current_step = 0
        
    def backward(self, loss):
        scaled_loss = loss / self.steps
        scaled_loss.backward()
        self.current_step += 1
        if self.current_step >= self.steps:
            self.optimizer_step()
            
    def optimizer_step(self):
        if self.current_step > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0

class ClassPerformanceMonitor:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
        self.class_losses = torch.zeros(self.num_classes)
        self.running_conf_matrix = torch.zeros((self.num_classes, self.num_classes))
        
    def update(self, outputs, targets, loss_per_sample):
        # Update per-class accuracy
        predictions = outputs.argmax(dim=1)
        for cls in range(self.num_classes):
            cls_mask = targets == cls
            self.class_correct[cls] += (predictions[cls_mask] == targets[cls_mask]).sum().item()
            self.class_total[cls] += cls_mask.sum().item()
            if cls_mask.any():
                self.class_losses[cls] += loss_per_sample[cls_mask].sum().item()
                
        # Update confusion matrix
        batch_conf = confusion_matrix(
            targets.cpu().numpy(), 
            predictions.cpu().numpy(),
            labels=range(self.num_classes)
        )
        self.running_conf_matrix += torch.tensor(batch_conf)
    
    def get_metrics(self):
        # Calculate per-class metrics
        class_acc = (self.class_correct / (self.class_total + 1e-8)).tolist()
        class_avg_loss = (self.class_losses / (self.class_total + 1e-8)).tolist()
        
        # Calculate per-class precision and recall from confusion matrix
        conf_matrix = self.running_conf_matrix.float()
        precision = torch.diag(conf_matrix) / (conf_matrix.sum(dim=0) + 1e-8)
        recall = torch.diag(conf_matrix) / (conf_matrix.sum(dim=1) + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'per_class_accuracy': class_acc,
            'per_class_loss': class_avg_loss,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'confusion_matrix': self.running_conf_matrix.tolist()
        }

def train_epoch(model, train_loader, criterion, optimizer, scheduler, config, 
                epoch=None, ema=None, monitor=None):
    model.train()
    perf_monitor = ClassPerformanceMonitor(config.num_classes)
    
    train_stats = {
        'loss': 0,
        'correct': 0,
        'total': 0
    }
    
    # Memory cleanup
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    train_pbar = tqdm(
        train_loader,
        desc=f"Training (Epoch {epoch})" if epoch is not None else "Training",
        leave=True,
        dynamic_ncols=True,
        position=0,
        file=sys.stdout
    )
    
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() and config.use_amp else None
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(train_pbar):
        batch_start = time.time()
        
        try:
            # Handle mixed batches (Mixup/Cutmix)
            if 'grade_a' in batch and 'grade_b' in batch:
                images = batch['image'].to(config.device, non_blocking=True)
                labels_a = batch['grade_a'].to(config.device, non_blocking=True)
                labels_b = batch['grade_b'].to(config.device, non_blocking=True)
                lam = batch['lam']
                is_mixed = True
            else:
                images = batch['image'].to(config.device, non_blocking=True)
                labels_a = batch['grade'].to(config.device, non_blocking=True)
                labels_b = labels_a
                lam = 1.0
                is_mixed = False
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type='mps' if torch.mps.is_available() else 'cpu') if config.use_amp else nullcontext():
                outputs = model(images)
                
                # Calculate loss with mixed batch handling
                if is_mixed:
                    loss = lam * criterion(outputs, labels_a) + \
                           (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels_a)
                
                # Get per-sample losses for monitoring
                with torch.no_grad():
                    if isinstance(outputs, tuple) and len(outputs) >= 4:
                        _, cls_out, _, _, _ = outputs
                    else:
                        _, cls_out, _ = outputs
                    logits = F.log_softmax(cls_out, dim=1)
                    loss_per_sample = F.nll_loss(logits, labels_a, reduction='none')
                
                loss = loss / config.gradient_accumulation_steps

            # Backward pass with gradient scaling
            if scaler:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Update schedulers and EMA
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if scheduler:
                    scheduler.step()
                if ema:
                    ema.update()
            
            # Update statistics
            train_stats['loss'] += loss.item() * config.gradient_accumulation_steps
            
            with torch.no_grad():
                target = labels_a if not is_mixed or lam > 0.5 else labels_b
                perf_monitor.update(cls_out, target, loss_per_sample)
            
            # Update progress bar
            metrics = perf_monitor.get_metrics()
            avg_loss = train_stats['loss'] / (batch_idx + 1)
            
            train_pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'min_cls_acc': f"{min(metrics['per_class_accuracy']):.4f}"
            }, refresh=True)
            
            # Log batch metrics
            if monitor:
                batch_time = time.time() - batch_start
                monitor.log_batch(
                    batch_idx=batch_idx,
                    loss=avg_loss,
                    per_class_acc=metrics['per_class_accuracy'],
                    lr=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time
                )
            
        except RuntimeError as e:
            print(f"\nError during training: {str(e)}")
            if "out of memory" in str(e):
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                return None
            raise e
        
        # Memory cleanup
        del images, labels_a, labels_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    train_pbar.close()
    
    # Get final metrics
    final_metrics = perf_monitor.get_metrics()
    metrics = {
        'loss': train_stats['loss'] / len(train_loader),
        'accuracy': final_metrics['accuracy'],
        'per_class_accuracy': final_metrics['per_class_accuracy'],
        'per_class_loss': final_metrics['per_class_loss'],
        'per_class_precision': final_metrics['per_class_precision'],
        'per_class_recall': final_metrics['per_class_recall'],
        'per_class_f1': final_metrics['per_class_f1'],
        'confusion_matrix': final_metrics['confusion_matrix']
    }
    
    return metrics

def validate_epoch(model, val_loader, criterion, config, monitor=None, epoch=None, fold=None):
    """
    Validation function focused purely on classification metrics.
    """
    model.eval()
    val_stats = {
        'loss': 0,
        'correct': 0,
        'total': 0
    }
    
    # For storing predictions and ground truth
    y_true, y_pred = [], []
    val_pbar = tqdm(
        val_loader,
        desc=f"Validation (Epoch {epoch}, Fold {fold})" if epoch is not None and fold is not None else "Validation",
        leave=True,
        dynamic_ncols=True,
        position=0,
        file=sys.stdout
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            images = batch['image'].to(config.device, non_blocking=True)
            grades = batch['grade'].to(config.device, non_blocking=True)
            
            with torch.amp.autocast(
                device_type='mps' if torch.mps.is_available() else 'cpu'
            ) if config.use_amp else nullcontext():
                # Get model outputs
                outputs = model(images)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) >= 4:
                    _, cls_out, _, _, _ = outputs  # Training mode outputs
                else:
                    _, cls_out, _ = outputs  # Inference mode outputs
                
                # Compute loss
                loss = criterion(outputs, grades)
                
                # Update statistics
                val_stats['loss'] += loss.item()
                predictions = cls_out.argmax(1)
                val_stats['correct'] += (predictions == grades).sum().item()
                val_stats['total'] += grades.size(0)
                
                # Store predictions for metrics
                y_true.extend(grades.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                
                # Update progress bar
                avg_loss = val_stats['loss'] / (batch_idx + 1)
                accuracy = val_stats['correct'] / val_stats['total']
                val_pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'acc': f"{accuracy:.4f}"
                }, refresh=True)
            
            # Clean up memory
            del images, grades, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
    
    val_pbar.close()
    
    # Compute final metrics
    metrics = {
        'loss': val_stats['loss'] / len(val_loader),
        'accuracy': val_stats['correct'] / val_stats['total'],
        'per_class_acc': compute_per_class_accuracy(y_true, y_pred, config.num_classes),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'true_labels': y_true,
        'pred_labels': y_pred
    }
    
    # Print per-class accuracies
    print("\nPer-class Accuracies:")
    for class_idx, acc in enumerate(metrics['per_class_acc']):
        print(f" Class {class_idx}: {acc:.4f}")
    
    return metrics

def compute_per_class_accuracy(y_true, y_pred, num_classes):
    """Compute accuracy for each class"""
    per_class_acc = []
    for i in range(num_classes):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    return per_class_acc


# Utility functions
def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def track_memory():
    """
    Track current GPU or MPS memory usage in MB
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    elif hasattr(torch.mps, 'current_allocated_memory'):
        return torch.mps.current_allocated_memory() / 1024**2
    return 0

def train_with_cross_validation(config):
    """
    Cross-validation training loop optimized for classification only.
    """
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    dataset = KneeDataset(config.train_dir, phase='train')
    labels = [sample['grade'] for sample in dataset.samples]
    
    monitor = TrainingMonitor(config.output_dir)
    
    # Get class counts for loss weighting
    class_counts = [sum(1 for label in labels if label == i) for i in range(config.num_classes)]
    
    current_batch_size = config.batch_size
    
    # Initialize criterion with class weights
    criterion = ClassificationOnlyLoss(class_counts, device='mps')


    # Initialize for LR finding
    temp_model = HybridModel(num_classes=config.num_classes).to(config.device)
    temp_optimizer = torch.optim.AdamW(
        temp_model.parameters(),
        lr=config.max_lr,
        weight_decay=config.weight_decay
    )
    
    # Create temporary dataloader for LR finding
    temp_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # multiprocessing_context='spawn'
    )
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    optimal_lr = find_lr_for_grayscale(
        model=temp_model,
        train_loader=temp_loader,
        criterion=criterion,
        optimizer=temp_optimizer,
        device=config.device,
        num_iter=100,
        start_lr=1e-5,  # Modified to start higher
        end_lr=1e-2     # Modified to end lower
    )
    
    print(f"Optimal learning rate found: {optimal_lr:.2e}")
    config.max_lr = optimal_lr
    
    # Clean up temporary objects
    del temp_model, temp_optimizer, temp_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nStarting Fold {fold + 1}/{config.num_folds}")
        monitor.log_scalar('fold_start', fold, step=fold)
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Initialize with starting size
        current_size = config.size_schedule[0][1]
        dataset.update_transforms(current_size)
        
        # Initialize augmentation pipeline
        augmentor = ClassificationAugmentor(config)
        
        # Update dataset transforms
        dataset.transform = augmentor.train_transform
        
        # Use weighted sampler for training
        train_sampler = create_balanced_sampler(dataset, train_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=current_batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        model = HybridModel(num_classes=config.num_classes).to(config.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr/config.div_factor,
            weight_decay=config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=config.div_factor,
            final_div_factor=1e4
        )
        
        ema = EMA(model, decay=0.999)
        scaler = torch.amp.GradScaler(enabled=config.use_amp)
        
        best_val_metric = 0
        patience_counter = 0
        previous_accuracy = 0.0
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            # Check for image size updates
            for schedule_epoch, new_size in config.size_schedule:
                if epoch == schedule_epoch and new_size != current_size:
                    current_size = new_size
                    dataset.update_transforms(current_size)
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=current_batch_size,
                        sampler=train_sampler,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory
                    )
                    
            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                monitor=monitor,
                config=config,
                epoch=epoch,
                ema=ema
            )
            
            if epoch % config.val_freq == 0:
                ema.apply_shadow()
                val_metrics = validate_epoch(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    monitor=monitor,
                    epoch=epoch,
                    fold=fold
                )
                ema.restore()
                
                # Model selection based on balanced accuracy
                balanced_acc = np.mean(list(val_metrics['per_class_acc'].values()))
                monitor.log_scalar('balanced_accuracy', balanced_acc, step=epoch, fold=str(fold))
                
                if balanced_acc > best_val_metric:
                    best_val_metric = balanced_acc
                    patience_counter = 0
                    save_path = config.model_dir / f'best_model_fold_{fold}.pt'
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'metrics': val_metrics,
                        'config': vars(config)
                    }, save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    print("\nEarly stopping triggered!")
                    break
            
            # Dynamic batch size adjustment
            if (train_metrics['accuracy'] > previous_accuracy + config.min_improvement 
                and current_batch_size < config.max_batch_size):
                new_batch_size = min(current_batch_size * 2, config.max_batch_size)
                if new_batch_size > current_batch_size:
                    current_batch_size = new_batch_size
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=current_batch_size,
                        sampler=train_sampler,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory
                    )
            
            previous_accuracy = train_metrics['accuracy']
        
        fold_metrics.append({
            'fold': fold,
            'best_balanced_acc': best_val_metric
        })
        
        # Cleanup
        del model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    return fold_metrics



if __name__ == "__main__":
    train_with_cross_validation(config)