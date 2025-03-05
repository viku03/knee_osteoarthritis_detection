import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

class TrainingMonitor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / 'logs'
        self.plot_dir = self.output_dir / 'plots'
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.model_dir = self.output_dir / 'models'
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Ensure directories exist
        for dir_path in [self.log_dir, self.plot_dir, self.checkpoint_dir, self.model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Add debug flag
        self.debug = True
        
        # Initialize metrics with additional loss tracking
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_seg_loss': [], 'val_seg_loss': [],
            'train_cls_loss': [], 'val_cls_loss': [],
            'train_acc': [], 'val_acc': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'per_class_acc': {i: [] for i in range(5)},
            'learning_rates': [],
            'grad_norms': [],
            'batch_times': [],
            'epoch_times': [],
            'scalars': {}
        }
        
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def log_scalar(self, name, value, step, fold=None):
        """
        Log a scalar value with optional fold specification
        
        Args:
            name (str): Name of the scalar metric
            value (float): Value to log
            step (int): Current step/iteration
            fold (str, optional): Fold identifier for cross-validation
        """
        key = f"{name}/{fold}" if fold is not None else name
        if key not in self.metrics['scalars']:
            self.metrics['scalars'][key] = []
            
        scalar_data = {
            'step': step,
            'value': float(value),
            'timestamp': time.time()
        }
        
        self.metrics['scalars'][key].append(scalar_data)
        
        # Save to disk
        scalar_log_file = self.log_dir / 'scalar_metrics.jsonl'
        with open(scalar_log_file, 'a') as f:
            log_entry = {
                'name': name,
                'fold': fold,
                **scalar_data
            }
            f.write(json.dumps(log_entry) + '\n')
    
    def log_batch(self, batch_idx, loss, acc=None, accuracy=None, grad_norm=None, lr=None, 
                 batch_time=None, seg_loss=None, cls_loss=None):
        """Log metrics for each batch with flexible accuracy parameter naming"""
        if grad_norm is not None:
            self.metrics['grad_norms'].append(grad_norm)
        if batch_time is not None:
            self.metrics['batch_times'].append(batch_time)
        if lr is not None:
            self.metrics['learning_rates'].append(lr)
        
        # Use accuracy if acc is not provided
        final_acc = acc if acc is not None else accuracy
        
        if batch_idx % 50 == 0:
            self._save_batch_metrics(batch_idx, loss, final_acc, grad_norm, lr, seg_loss, cls_loss)

    def _get_metric_value(self, metrics, key_options):
        """Helper method to get metric value with multiple possible keys"""
        for key in key_options:
            if key in metrics:
                return metrics[key]
        raise KeyError(f"None of the keys {key_options} found in metrics")
    
    def log_epoch(self, epoch, train_metrics, val_metrics, epoch_time=None):
        """Log metrics for each epoch with flexible key names"""
        try:
            # Handle both 'acc' and 'accuracy' keys
            train_acc = self._get_metric_value(train_metrics, ['acc', 'accuracy'])
            val_acc = self._get_metric_value(val_metrics, ['acc', 'accuracy'])
            
            # Store basic metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            
            # Store segmentation and classification losses if available
            if 'seg_loss' in train_metrics:
                self.metrics['train_seg_loss'].append(train_metrics['seg_loss'])
            if 'seg_loss' in val_metrics:
                self.metrics['val_seg_loss'].append(val_metrics['seg_loss'])
            if 'cls_loss' in train_metrics:
                self.metrics['train_cls_loss'].append(train_metrics['cls_loss'])
            if 'cls_loss' in val_metrics:
                self.metrics['val_cls_loss'].append(val_metrics['cls_loss'])
            if 'f1_score' in val_metrics:
                self.metrics['f1_scores'].append(val_metrics['f1_score'])
            if 'precision' in val_metrics:
                self.metrics['precision_scores'].append(val_metrics['precision'])
            if 'recall' in val_metrics:
                self.metrics['recall_scores'].append(val_metrics['recall'])
            
            if epoch_time is not None:
                self.metrics['epoch_times'].append(epoch_time)
            
            # Store per-class accuracies
            if 'per_class_acc' in val_metrics:
                for class_idx, acc in val_metrics['per_class_acc'].items():
                    self.metrics['per_class_acc'][class_idx].append(acc)
            
            # Save metrics and create plots
            self._save_epoch_metrics(epoch)
            self._create_plots(epoch)
            
            # Check for best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                return True
            return False
            
        except KeyError as e:
            print(f"Error logging epoch metrics: {str(e)}")
            print("Available train metrics keys:", train_metrics.keys())
            print("Available validation metrics keys:", val_metrics.keys())
            raise
    
    def _save_batch_metrics(self, batch_idx, loss, acc, grad_norm=None, lr=None, 
                          seg_loss=None, cls_loss=None):
        """Save batch metrics with additional loss components"""
        batch_data = {
            'batch_idx': batch_idx,
            'loss': float(loss),
            'acc': float(acc),
            'timestamp': time.time()
        }
        
        if grad_norm is not None:
            batch_data['grad_norm'] = float(grad_norm)
        if lr is not None:
            batch_data['lr'] = float(lr)
        if seg_loss is not None:
            batch_data['seg_loss'] = float(seg_loss)
        if cls_loss is not None:
            batch_data['cls_loss'] = float(cls_loss)
        
        with open(self.log_dir / 'batch_metrics.jsonl', 'a') as f:
            f.write(json.dumps(batch_data) + '\n')
    
    def _save_epoch_metrics(self, epoch):
        """Save epoch metrics including new performance metrics"""
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(self.metrics['train_loss'][-1]),
            'val_loss': float(self.metrics['val_loss'][-1]),
            'train_acc': float(self.metrics['train_acc'][-1]),
            'val_acc': float(self.metrics['val_acc'][-1]),
            'timestamp': time.time()
        }
        
        # Add new metrics if available
        if self.metrics['f1_scores']:
            epoch_data['f1_score'] = float(self.metrics['f1_scores'][-1])
        if self.metrics['precision_scores']:
            epoch_data['precision'] = float(self.metrics['precision_scores'][-1])
        if self.metrics['recall_scores']:
            epoch_data['recall'] = float(self.metrics['recall_scores'][-1])
        if self.metrics['train_seg_loss']:
            epoch_data['train_seg_loss'] = float(self.metrics['train_seg_loss'][-1])
        if self.metrics['val_seg_loss']:
            epoch_data['val_seg_loss'] = float(self.metrics['val_seg_loss'][-1])
        if self.metrics['train_cls_loss']:
            epoch_data['train_cls_loss'] = float(self.metrics['train_cls_loss'][-1])
        if self.metrics['val_cls_loss']:
            epoch_data['val_cls_loss'] = float(self.metrics['val_cls_loss'][-1])
        
        if self.metrics['per_class_acc']:
            epoch_data['per_class_acc'] = {
                k: float(v[-1]) for k, v in self.metrics['per_class_acc'].items()
            }
        
        with open(self.log_dir / 'epoch_metrics.json', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')
    
    def _create_plots(self, epoch):
        self._plot_loss_curve(epoch)
        self._plot_accuracy_curve(epoch)
        self._plot_per_class_accuracy(epoch)
        self._plot_gradient_norms(epoch)
    
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
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.plot_dir / f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
        
    def save_classification_report(self, y_true, y_pred, epoch):
        report = classification_report(y_true, y_pred)
        with open(self.log_dir / f'classification_report_epoch_{epoch}.txt', 'w') as f:
            f.write(report)