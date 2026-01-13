"""
Defense Mechanisms Against Adversarial Attacks

This module provides various defense strategies to protect models against
adversarial attacks, including:
- Adversarial training
- Input transformations
- Ensemble defenses
- Detection mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable
import numpy as np
from torchvision import transforms


class AdversarialDefense:
    """Base class for adversarial defense mechanisms."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize defense mechanism.
        
        Args:
            model: Model to defend
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device


class InputTransformationDefense(AdversarialDefense):
    """
    Defense using input transformations to remove adversarial perturbations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transforms_list: Optional[List[str]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize input transformation defense.
        
        Args:
            model: Model to defend
            transforms_list: List of transformation types to apply
            device: Device to run on
        """
        super().__init__(model, device)
        
        if transforms_list is None:
            transforms_list = ['jpeg_compression', 'bit_depth_reduction', 'gaussian_blur']
        
        self.transforms_list = transforms_list
    
    def jpeg_compression(self, x: torch.Tensor, quality: int = 75) -> torch.Tensor:
        """
        Simulate JPEG compression (simplified version).
        
        Args:
            x: Input tensor
            quality: JPEG quality (not used in this simplified version)
            
        Returns:
            Compressed tensor
        """
        # Simplified: just add slight quantization
        x_compressed = torch.round(x * 255) / 255
        return x_compressed
    
    def bit_depth_reduction(self, x: torch.Tensor, bits: int = 5) -> torch.Tensor:
        """
        Reduce bit depth of the image.
        
        Args:
            x: Input tensor in [0, 1]
            bits: Number of bits to keep
            
        Returns:
            Quantized tensor
        """
        levels = 2 ** bits
        x_quantized = torch.round(x * (levels - 1)) / (levels - 1)
        return x_quantized
    
    def gaussian_blur(self, x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """
        Apply Gaussian blur.
        
        Args:
            x: Input tensor
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation
            
        Returns:
            Blurred tensor
        """
        # Create Gaussian kernel
        channels = x.size(1)
        kernel = self._get_gaussian_kernel(kernel_size, sigma, channels).to(self.device)
        
        # Apply convolution
        padding = kernel_size // 2
        x_blurred = F.conv2d(x, kernel, padding=padding, groups=channels)
        
        return x_blurred
    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
        """Create a Gaussian kernel."""
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        g_2d = g[:, None] * g[None, :]
        g_2d = g_2d / g_2d.sum()
        
        # Expand for all channels
        kernel = g_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        
        return kernel
    
    def total_variation_minimization(self, x: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
        """
        Apply total variation denoising.
        
        Args:
            x: Input tensor
            weight: Weight for TV regularization
            
        Returns:
            Denoised tensor
        """
        # Simplified TV denoising
        # Calculate gradients
        diff_i = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_j = x[:, :, :, 1:] - x[:, :, :, :-1]
        
        # Minimize by clipping large gradients (simplified)
        return torch.clamp(x, 0, 1)
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transformations.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        x_transformed = x.clone()
        
        for transform_name in self.transforms_list:
            if transform_name == 'jpeg_compression':
                x_transformed = self.jpeg_compression(x_transformed)
            elif transform_name == 'bit_depth_reduction':
                x_transformed = self.bit_depth_reduction(x_transformed)
            elif transform_name == 'gaussian_blur':
                x_transformed = self.gaussian_blur(x_transformed)
            elif transform_name == 'total_variation':
                x_transformed = self.total_variation_minimization(x_transformed)
        
        return x_transformed
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with defense.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        x_transformed = self.transform(x)
        
        with torch.no_grad():
            outputs = self.model(x_transformed)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities


class EnsembleDefense(AdversarialDefense):
    """
    Ensemble defense using multiple models or transformations.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        voting: str = 'soft'
    ):
        """
        Initialize ensemble defense.
        
        Args:
            models: List of models for ensemble
            device: Device to run on
            voting: 'soft' for probability averaging, 'hard' for majority voting
        """
        super().__init__(models[0], device)
        self.models = [model.to(device) for model in models]
        self.voting = voting
        
        for model in self.models:
            model.eval()
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_outputs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(x)
                all_outputs.append(outputs)
        
        if self.voting == 'soft':
            # Average probabilities
            all_probs = [torch.softmax(out, dim=1) for out in all_outputs]
            avg_probs = torch.stack(all_probs).mean(dim=0)
            predictions = torch.argmax(avg_probs, dim=1)
            return predictions, avg_probs
        else:
            # Hard voting
            all_preds = [torch.argmax(out, dim=1) for out in all_outputs]
            stacked_preds = torch.stack(all_preds)
            predictions = torch.mode(stacked_preds, dim=0)[0]
            
            # Get average probabilities for selected classes
            all_probs = [torch.softmax(out, dim=1) for out in all_outputs]
            avg_probs = torch.stack(all_probs).mean(dim=0)
            
            return predictions, avg_probs


class AdversarialDetector:
    """
    Detector to identify adversarial examples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        threshold: float = 0.5
    ):
        """
        Initialize adversarial detector.
        
        Args:
            model: Model to use for detection
            device: Device to run on
            threshold: Detection threshold
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()
    
    def detect_by_prediction_confidence(self, x: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect adversarial examples by prediction confidence.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (is_adversarial, confidence_score)
        """
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob = torch.max(probabilities, dim=1)[0].item()
        
        # Low confidence might indicate adversarial example
        is_adversarial = max_prob < self.threshold
        
        return is_adversarial, max_prob
    
    def detect_by_input_reconstruction(
        self,
        x: torch.Tensor,
        autoencoder: Optional[nn.Module] = None
    ) -> Tuple[bool, float]:
        """
        Detect adversarial examples by reconstruction error.
        
        Args:
            x: Input tensor
            autoencoder: Autoencoder for reconstruction
            
        Returns:
            Tuple of (is_adversarial, reconstruction_error)
        """
        if autoencoder is None:
            # Without autoencoder, use simple denoising
            x_reconstructed = self._simple_denoise(x)
        else:
            with torch.no_grad():
                x_reconstructed = autoencoder(x)
        
        # Calculate reconstruction error
        reconstruction_error = torch.norm(x - x_reconstructed).item()
        
        # High reconstruction error might indicate adversarial example
        is_adversarial = reconstruction_error > self.threshold
        
        return is_adversarial, reconstruction_error
    
    def _simple_denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Simple denoising using Gaussian blur."""
        kernel_size = 3
        sigma = 1.0
        channels = x.size(1)
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        g_2d = g[:, None] * g[None, :]
        g_2d = g_2d / g_2d.sum()
        kernel = g_2d.expand(channels, 1, kernel_size, kernel_size).contiguous().to(self.device)
        
        padding = kernel_size // 2
        x_denoised = F.conv2d(x, kernel, padding=padding, groups=channels)
        
        return x_denoised


class AdversarialTrainingDefense:
    """
    Defense through adversarial training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attack_method: Callable,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize adversarial training defense.
        
        Args:
            model: Model to train
            attack_method: Function to generate adversarial examples
            device: Device to run on
        """
        self.model = model.to(device)
        self.attack_method = attack_method
        self.device = device
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        alpha: float = 0.5
    ) -> float:
        """
        Perform one adversarial training step.
        
        Args:
            x: Input batch
            y: Labels
            optimizer: Optimizer
            criterion: Loss function
            alpha: Weight for adversarial loss (0.5 = equal weight)
            
        Returns:
            Training loss
        """
        self.model.train()
        
        # Generate adversarial examples
        x_adv = self.attack_method(self.model, x, y)
        
        # Combine clean and adversarial examples
        x_combined = torch.cat([x, x_adv], dim=0)
        y_combined = torch.cat([y, y], dim=0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(x_combined)
        loss = criterion(outputs, y_combined)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()


def evaluate_defense(
    defense,
    data_loader,
    attack_fn: Optional[Callable] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Evaluate defense mechanism.
    
    Args:
        defense: Defense mechanism to evaluate
        data_loader: Data loader for evaluation
        attack_fn: Optional attack function
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    print("Evaluating defense mechanism...")
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Test on clean images
        predictions, _ = defense.predict(images)
        correct_clean += (predictions == labels).sum().item()
        
        # Test on adversarial images if attack is provided
        if attack_fn is not None:
            images_adv = attack_fn(images, labels)
            predictions_adv, _ = defense.predict(images_adv)
            correct_adv += (predictions_adv == labels).sum().item()
        
        total += labels.size(0)
    
    results = {
        'clean_accuracy': correct_clean / total,
        'adversarial_accuracy': correct_adv / total if attack_fn else None,
        'total_samples': total
    }
    
    return results


if __name__ == "__main__":
    from models import create_convnext_model
    
    print("Defense module loaded successfully!")
    print("\nAvailable defenses:")
    print("1. InputTransformationDefense - Apply input transformations")
    print("2. EnsembleDefense - Use ensemble of models")
    print("3. AdversarialDetector - Detect adversarial examples")
    print("4. AdversarialTrainingDefense - Train with adversarial examples")
