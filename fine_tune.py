"""
Fine-tuning ConvNext Models with Warm-up and Advanced Training Features

This module provides utilities for fine-tuning ConvNext models with:
- Learning rate warm-up
- Cosine annealing scheduler
- Early stopping
- Model checkpointing
- Mixed precision training support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Callable
import os
import time
from tqdm import tqdm


class WarmupCosineScheduler:
    """Learning rate scheduler with warm-up and cosine annealing."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warm-up epochs
            total_epochs: Total number of training epochs
            base_lr: Base learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warm-up
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159265359))
            )
            lr = float(lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelTrainer:
    """Trainer class for fine-tuning ConvNext models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_mixed_precision: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.scaler = GradScaler() if self.use_mixed_precision else None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 5,
        weight_decay: float = 0.01,
        patience: int = 10,
        save_dir: str = './checkpoints',
        save_best_only: bool = True
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Base learning rate
            warmup_epochs: Number of warm-up epochs
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            base_lr=learning_rate
        )
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, mode='min')
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, criterion, epoch)
            
            # Update learning rate
            current_lr = scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
            print(f"LR: {current_lr:.6f}")
            print(f"Train Loss: {train_metrics['loss']:.4f} - Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if save_best_only:
                    save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy']
                    }, save_path)
                    print(f"Saved best model to {save_path}")
            
            # Check early stopping
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\nTraining completed!")
        return self.history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")


if __name__ == "__main__":
    from models import create_convnext_model
    
    print("Fine-tuning module loaded successfully!")
    print("\nExample usage:")
    print("model = create_convnext_model('convnext_tiny', pretrained=True, num_classes=10)")
    print("trainer = ModelTrainer(model)")
    print("history = trainer.fit(train_loader, val_loader, epochs=50)")
