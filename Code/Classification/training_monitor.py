import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class TrainingMonitor:
    def __init__(self, output_dir):
        try:
            self.output_dir = Path(output_dir)
            self.log_dir = self.output_dir / 'logs'
            self.plot_dir = self.output_dir / 'plots'
            self.checkpoint_dir = self.output_dir / 'checkpoints'
            self.model_dir = self.output_dir / 'models'
            self.attention_dir = self.output_dir / 'attention_maps'

            # Create directories with error handling
            for dir_path in [self.log_dir, self.plot_dir, self.checkpoint_dir, 
                           self.model_dir, self.attention_dir]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    raise RuntimeError(f"Permission denied when creating directory: {dir_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to create directory {dir_path}: {str(e)}")

            # Initialize metrics with type checking
            self.metrics = {
                'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': [],
                'f1_scores': [],
                'precision_scores': [],
                'recall_scores': [],
                'per_class_acc': {i: [] for i in range(5)},
                'learning_rates': [],
                'grad_norms': [],
                'batch_times': [],
                'epoch_times': [],
                'scalars': {},
                'attention_scores': [],
                'kappa_scores': [],
                'auc_roc_scores': [],
                'per_class_auc': {i: [] for i in range(5)}
            }

            self.best_val_acc = 0.0
            self.best_val_loss = float('inf')
            self.epochs_without_improvement = 0

        except Exception as e:
            raise RuntimeError(f"Failed to initialize TrainingMonitor: {str(e)}")

        # Define metric key mappings
        self.metric_mappings = {
            'loss': ['loss', 'total_loss'],  # Try these keys in order
            'acc': ['acc', 'accuracy'],
            'f1_score': ['f1_score', 'f1_weighted'],
            'precision': ['precision', 'precision_weighted'],
            'recall': ['recall', 'recall_weighted'],
            'kappa': ['kappa'],
            'auc_roc': ['auc_roc']
        }


    def log_scalar(self, name, value, step, fold=None):
        """Log a scalar value with enhanced error handling"""
        try:
            # Type checking
            if not isinstance(name, str):
                raise ValueError("Scalar name must be a string")
            if not isinstance(step, (int, np.integer)):
                raise ValueError("Step must be an integer")
            
            # Convert value to float and check for validity
            try:
                float_value = float(value)
                if not np.isfinite(float_value):
                    raise ValueError("Value must be finite")
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for scalar {name}: {value}")

            key = f"{name}/{fold}" if fold is not None else name
            if key not in self.metrics['scalars']:
                self.metrics['scalars'][key] = []

            scalar_data = {
                'step': step,
                'value': float_value,
                'timestamp': time.time()
            }

            self.metrics['scalars'][key].append(scalar_data)

            # Safe file writing
            scalar_log_file = self.log_dir / 'scalar_metrics.jsonl'
            try:
                with open(scalar_log_file, 'a') as f:
                    log_entry = {
                        'name': name,
                        'fold': fold,
                        **scalar_data
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except (IOError, PermissionError) as e:
                print(f"Warning: Failed to write scalar log to file: {str(e)}")

        except Exception as e:
            print(f"Error in log_scalar: {str(e)}")
            return None
    
    def log_batch(self, batch_idx, loss, acc=None, accuracy=None, grad_norm=None, 
                  lr=None, batch_time=None, seg_loss=None, cls_loss=None):
        """Log batch metrics with type checking and error handling"""
        try:
            # Type checking and validation
            if not isinstance(batch_idx, (int, np.integer)):
                raise ValueError("batch_idx must be an integer")
            
            # Convert and validate loss
            try:
                loss = float(loss)
                if not np.isfinite(loss):
                    raise ValueError("Loss must be finite")
            except (TypeError, ValueError):
                raise ValueError(f"Invalid loss value: {loss}")
            
            # Handle accuracy with flexible input
            final_acc = None
            if acc is not None:
                final_acc = float(acc)
            elif accuracy is not None:
                final_acc = float(accuracy)
            
            if final_acc is not None and (final_acc < 0 or final_acc > 1):
                raise ValueError("Accuracy must be between 0 and 1")

            # Update metrics
            if grad_norm is not None:
                grad_norm = float(grad_norm)
                if np.isfinite(grad_norm):
                    self.metrics['grad_norms'].append(grad_norm)
                
            if batch_time is not None:
                batch_time = float(batch_time)
                if batch_time > 0:
                    self.metrics['batch_times'].append(batch_time)
                
            if lr is not None:
                lr = float(lr)
                if lr > 0:
                    self.metrics['learning_rates'].append(lr)

            # Log metrics if appropriate
            if batch_idx % 50 == 0:
                self._save_batch_metrics(batch_idx, loss, final_acc, grad_norm, lr, 
                                      seg_loss, cls_loss)

        except Exception as e:
            print(f"Error in log_batch: {str(e)}")
    
    def _save_plots_safely(self, plot_func, filename, *args, **kwargs):
        """Safely save plots with error handling"""
        try:
            plt.figure(figsize=(10, 6))
            plot_func(*args, **kwargs)
            plt.savefig(self.plot_dir / filename)
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to save plot {filename}: {str(e)}")
            try:
                plt.close()  # Always try to close the figure
            except:
                pass
            
    def _get_metric_value_safe(self, metrics_dict, key_options, default_value=0.0):
        """Helper method to safely get metric value with multiple possible keys and default handling"""
        try:
            for key in key_options:
                if key in metrics_dict:
                    value = metrics_dict[key]
                    return default_value if value is None else value
            return default_value
        except Exception:
            return default_value

    def _safe_append_metric(self, metric_list_name, metrics_dict, key_options, default_value=0.0):
        """Safely append a metric to its list with default handling"""
        try:
            value = self._get_metric_value_safe(metrics_dict, key_options, default_value)
            self.metrics[metric_list_name].append(value)
        except Exception:
            pass

    
    def log_epoch(self, epoch, train_metrics, val_metrics, epoch_time=None):
        """Log metrics for each epoch with enhanced logging and key mapping"""
        try:
            # Get training metrics using mappings with default handling
            train_loss = self._get_metric_value_safe(train_metrics, self.metric_mappings['loss'], 0.0)
            train_acc = self._get_metric_value_safe(train_metrics, self.metric_mappings['acc'], 0.0)
            
            # Get validation metrics using mappings with default handling
            val_loss = self._get_metric_value_safe(val_metrics, self.metric_mappings['loss'], train_loss)
            val_acc = self._get_metric_value_safe(val_metrics, self.metric_mappings['acc'], 0.0)
            
            # Store basic metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            
            if epoch_time is not None:
                self.metrics['epoch_times'].append(epoch_time)
            
            # Store additional metrics if available
            self._safe_append_metric('f1_scores', val_metrics, self.metric_mappings['f1_score'])
            self._safe_append_metric('precision_scores', val_metrics, self.metric_mappings['precision'])
            self._safe_append_metric('recall_scores', val_metrics, self.metric_mappings['recall'])
            
            # Store per-class accuracies if available
            if 'per_class_acc' in val_metrics:
                for class_idx, acc in val_metrics['per_class_acc'].items():
                    if class_idx not in self.metrics['per_class_acc']:
                        self.metrics['per_class_acc'][class_idx] = []
                    self.metrics['per_class_acc'][class_idx].append(acc)
            
            # Store medical metrics if available
            self._safe_append_metric('kappa_scores', val_metrics, self.metric_mappings['kappa'])
            self._safe_append_metric('auc_roc_scores', val_metrics, self.metric_mappings['auc_roc'])
            
            # Print epoch summary with safe formatting
            print(f"\nEpoch {epoch} Summary:")
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
            # Save metrics and create plots
            self._save_epoch_metrics(epoch)
            self._create_plots(epoch)
            
            # Check for best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print("\n*** New Best Validation Accuracy! ***")
                return True
                
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("\n*** New Best Validation Loss! ***")
            
            return False
            
        except Exception as e:
            print(f"Error in log_epoch: {str(e)}")
            print("Available train metrics keys:", train_metrics.keys())
            print("Available validation metrics keys:", val_metrics.keys())
            raise


    
    def _save_batch_metrics(self, batch_idx, loss, acc, grad_norm=None, lr=None, 
                          seg_loss=None, cls_loss=None):
        """Save batch metrics with safe file handling"""
        try:
            batch_data = {
                'batch_idx': int(batch_idx),
                'loss': float(loss),
                'timestamp': time.time()
            }

            if acc is not None:
                batch_data['acc'] = float(acc)
            if grad_norm is not None:
                batch_data['grad_norm'] = float(grad_norm)
            if lr is not None:
                batch_data['lr'] = float(lr)
            if seg_loss is not None:
                batch_data['seg_loss'] = float(seg_loss)
            if cls_loss is not None:
                batch_data['cls_loss'] = float(cls_loss)

            # Safe file writing with error handling
            try:
                with open(self.log_dir / 'batch_metrics.jsonl', 'a') as f:
                    f.write(json.dumps(batch_data) + '\n')
            except (IOError, PermissionError) as e:
                print(f"Warning: Failed to write batch metrics to file: {str(e)}")

        except Exception as e:
            print(f"Error in _save_batch_metrics: {str(e)}")

    
    def _save_epoch_metrics(self, epoch):
        """Save epoch metrics with safe handling of per-class accuracies"""
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(self.metrics['train_loss'][-1]),
            'val_loss': float(self.metrics['val_loss'][-1]),
            'train_acc': float(self.metrics['train_acc'][-1]),
            'val_acc': float(self.metrics['val_acc'][-1]),
            'timestamp': time.time()
        }
        
        # Add per-class accuracies if available
        if self.metrics['per_class_acc']:
            per_class_data = {}
            for class_idx, accuracies in self.metrics['per_class_acc'].items():
                if accuracies:  # Only add if there are values
                    per_class_data[str(class_idx)] = float(accuracies[-1])
            if per_class_data:
                epoch_data['per_class_acc'] = per_class_data
        
        # Add other metrics if available
        if self.metrics['f1_scores']:
            epoch_data['f1_score'] = float(self.metrics['f1_scores'][-1])
        if self.metrics['precision_scores']:
            epoch_data['precision'] = float(self.metrics['precision_scores'][-1])
        if self.metrics['recall_scores']:
            epoch_data['recall'] = float(self.metrics['recall_scores'][-1])
        
        # Save to file
        with open(self.log_dir / 'epoch_metrics.jsonl', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')
    
    def _create_plots(self, epoch):
        """Create all plots with error handling"""
        plot_functions = [
            (self._plot_loss_curve, 'loss'),
            (self._plot_accuracy_curve, 'accuracy'),
            (self._plot_per_class_accuracy, 'per_class_accuracy'),
            (self._plot_gradient_norms, 'gradient_norms'),
            (self._plot_attention_metrics, 'attention_metrics'),
            (self._plot_medical_metrics, 'medical_metrics'),
            (self._plot_roc_curves, 'roc_curves')
        ]

        for plot_func, plot_name in plot_functions:
            try:
                plot_func(epoch)
            except Exception as e:
                print(f"Warning: Failed to create {plot_name} plot: {str(e)}")
    
    def _plot_loss_curve(self, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.plot_dir / f'loss_epoch_{epoch}.png')
        plt.close()
    
    def _plot_accuracy_curve(self, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['train_acc'], label='Train Acc')
        plt.plot(self.metrics['val_acc'], label='Val Acc')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.plot_dir / f'acc_epoch_{epoch}.png')
        plt.close()
    
    def _plot_per_class_accuracy(self, epoch):
        plt.figure(figsize=(12, 6))
        for class_idx in range(5):
            plt.plot(self.metrics['per_class_acc'][class_idx], 
                    label=f'Class {class_idx}')
        plt.title('Per-class Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.plot_dir / f'per_class_acc_epoch_{epoch}.png')
        plt.close()
    
    def _plot_gradient_norms(self, epoch):
        plt.figure(figsize=(8, 5))
        plt.hist(self.metrics['grad_norms'][-100:], bins=30)
        plt.title('Recent Gradient Norm Distribution')
        plt.xlabel('Gradient Norm')
        plt.ylabel('Count')
        plt.savefig(self.plot_dir / f'grad_norms_epoch_{epoch}.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix with input validation"""
        try:
            # Input validation
            if not isinstance(y_true, (np.ndarray, list)):
                raise ValueError("y_true must be a numpy array or list")
            if not isinstance(y_pred, (np.ndarray, list)):
                raise ValueError("y_pred must be a numpy array or list")
            
            # Convert to numpy arrays if needed
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Check shapes
            if y_true.shape != y_pred.shape:
                raise ValueError("y_true and y_pred must have the same shape")
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            self._save_plots_safely(
                sns.heatmap,
                f'confusion_matrix_epoch_{epoch}.png',
                cm, annot=True, fmt='d', cmap='Blues'
            )

        except Exception as e:
            print(f"Error in plot_confusion_matrix: {str(e)}")
        
    def save_classification_report(self, y_true, y_pred, epoch):
        report = classification_report(y_true, y_pred)
        with open(self.log_dir / f'classification_report_epoch_{epoch}.txt', 'w') as f:
            f.write(report)

    def _plot_attention_metrics(self, epoch):
        """Plot attention metrics over time"""
        if len(self.metrics['attention_scores']) > 0:
            plt.figure(figsize=(10, 6))
            
            epochs = [m['epoch'] for m in self.metrics['attention_scores']]
            means = [m['mean'] for m in self.metrics['attention_scores']]
            stds = [m['std'] for m in self.metrics['attention_scores']]
            
            plt.plot(epochs, means, label='Mean Attention')
            plt.fill_between(epochs, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2)
            
            plt.title('Attention Score Distribution Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Attention Score')
            plt.legend()
            plt.savefig(self.plot_dir / f'attention_metrics_epoch_{epoch}.png')
            plt.close()
    
    def plot_attention_maps(self, attention_maps, images, predictions, targets, epoch, batch_idx):
        """Modified for better X-ray visualization"""
        fig, axes = plt.subplots(2, attention_maps.size(0), figsize=(15, 8))
        
        for i in range(attention_maps.size(0)):
            # Plot original X-ray with proper medical imaging colormap
            img = images[i, 0].cpu().numpy()  # Take first channel only
            axes[0, i].imshow(img, cmap='bone')  # Use bone colormap for X-rays
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Pred: {predictions[i]}\nTrue: {targets[i]}')
            
            # Plot attention map with proper normalization
            attn = attention_maps[i].mean(0).cpu().numpy()
            axes[1, i].imshow(attn, cmap='hot')  # Use hot colormap for attention
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attention_epoch_{epoch}_batch_{batch_idx}.png')
        plt.close()

    def log_attention_metrics(self, attention_maps, epoch):
        """
        Log attention map statistics for analysis.
        
        Args:
            attention_maps (torch.Tensor): Attention maps from the model
            epoch (int): Current epoch
        """
        # Calculate attention statistics
        attention_stats = {
            'mean': float(attention_maps.mean().cpu().numpy()),
            'std': float(attention_maps.std().cpu().numpy()),
            'max': float(attention_maps.max().cpu().numpy()),
            'min': float(attention_maps.min().cpu().numpy())
        }
        
        # Log to metrics
        self.metrics['attention_scores'].append({
            'epoch': epoch,
            **attention_stats
        })
        
        # Save to disk
        with open(self.log_dir / 'attention_metrics.jsonl', 'a') as f:
            f.write(json.dumps({
                'epoch': epoch,
                **attention_stats
            }) + '\n')
    
    def _plot_medical_metrics(self, epoch):
        """Plot medical-specific metrics over time"""
        plt.figure(figsize=(12, 6))
        
        if self.metrics['kappa_scores']:
            plt.plot(self.metrics['kappa_scores'], label="Cohen's Kappa")
        if self.metrics['auc_roc_scores']:
            plt.plot(self.metrics['auc_roc_scores'], label='AUC-ROC')
        
        plt.title('Medical Metrics vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_dir / f'medical_metrics_epoch_{epoch}.png')
        plt.close()
    
    def _plot_roc_curves(self, epoch):
        """Plot ROC curves for each class"""
        if hasattr(self, '_latest_roc_data'):
            plt.figure(figsize=(10, 10))
            
            for class_idx, (fpr, tpr, auc) in self._latest_roc_data.items():
                plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves by Class')
            plt.legend(loc="lower right")
            plt.savefig(self.plot_dir / f'roc_curves_epoch_{epoch}.png')
            plt.close()
