# # from model_arch import HybridModel
# # def count_parameters(model):
# #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # model = HybridModel(num_classes=5)
# # print(f"Total trainable parameters: {count_parameters(model):,}")

# # from tqdm.auto import tqdm
# # import time
# # import sys

# # def test_tqdm():
# #     # Test 1: Basic tqdm
# #     print("Testing basic tqdm:")
# #     for _ in tqdm(range(10), desc="Basic test"):
# #         time.sleep(0.5)
    
# #     print("\nTesting tqdm with explicit parameters:")
# #     # Test 2: Tqdm with explicit parameters
# #     for _ in tqdm(
# #         range(10),
# #         desc="Parameter test",
# #         file=sys.stdout,
# #         dynamic_ncols=True,
# #         leave=True
# #     ):
# #         time.sleep(0.5)
    
# #     print("\nTesting manual carriage return:")
# #     # Test 3: Manual progress without tqdm
# #     total = 10
# #     for i in range(total):
# #         print(f"\rProgress: {i+1}/{total}", end='', flush=True)
# #         time.sleep(0.5)
# #     print()  # New line at end

# # if __name__ == "__main__":
# #     test_tqdm()

# import os
# from pathlib import Path
# output_dir = Path('outputs/run_1')
# os.makedirs(output_dir, exist_ok=True)
# print(f"Output directory exists: {output_dir.exists()}")
# print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")

import numpy as np
import random

# Assuming TrainingMonitor class is in the same directory
from pathlib import Path
from trainingMonitor import TrainingMonitor

# Initialize TrainingMonitor
output_dir = './test_monitor'
monitor = TrainingMonitor(output_dir=output_dir)

# Generate dummy data for testing
epochs = 5
steps_per_epoch = 10
num_classes = 5

for epoch in range(epochs):
    # Simulate training and validation metrics
    train_loss = random.uniform(0.5, 1.0) / (epoch + 1)
    val_loss = random.uniform(0.5, 1.0) / (epoch + 1.5)
    train_acc = random.uniform(0.6, 0.9) + epoch * 0.02
    val_acc = random.uniform(0.6, 0.9) + epoch * 0.01
    train_seg_loss = train_loss * 0.5
    val_seg_loss = val_loss * 0.5
    train_cls_loss = train_loss * 0.5
    val_cls_loss = val_loss * 0.5
    
    # Simulate per-class accuracy
    per_class_acc = {i: random.uniform(0.6, 0.9) for i in range(num_classes)}
    
    # Log batch metrics
    for batch_idx in range(steps_per_epoch):
        monitor.log_batch(
            batch_idx=batch_idx,
            loss=random.uniform(0.5, 1.0),
            acc=random.uniform(0.6, 0.9),
            grad_norm=random.uniform(0.01, 0.1),
            lr=0.001,
            batch_time=random.uniform(0.01, 0.1),
            seg_loss=random.uniform(0.1, 0.5),
            cls_loss=random.uniform(0.1, 0.5),
        )

    # Log epoch metrics
    train_metrics = {
        'loss': train_loss,
        'acc': train_acc,
        'seg_loss': train_seg_loss,
        'cls_loss': train_cls_loss
    }
    val_metrics = {
        'loss': val_loss,
        'acc': val_acc,
        'seg_loss': val_seg_loss,
        'cls_loss': val_cls_loss,
        'per_class_acc': per_class_acc,
        'f1_score': random.uniform(0.6, 0.9),
        'precision': random.uniform(0.6, 0.9),
        'recall': random.uniform(0.6, 0.9),
    }

    monitor.log_epoch(epoch, train_metrics, val_metrics, epoch_time=random.uniform(30, 60))

    # Simulate confusion matrix and classification report
    y_true = np.random.randint(0, num_classes, 100)
    y_pred = np.random.randint(0, num_classes, 100)

    monitor.plot_confusion_matrix(y_true, y_pred, epoch)
    monitor.save_classification_report(y_true, y_pred, epoch)

# Print directory contents for verification
print(f"Test outputs saved to: {Path(output_dir).absolute()}")
print("Generated files:")
for path in Path(output_dir).rglob("*"):
    print(path)