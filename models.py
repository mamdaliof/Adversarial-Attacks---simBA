"""
ConvNext Model Definitions and Loading Utilities

This module provides utilities for loading and managing ConvNext models
for adversarial attack experiments.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


class ConvNextModelLoader:
    """Utility class for loading ConvNext models."""
    
    AVAILABLE_MODELS = {
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
    }
    
    @staticmethod
    def load_model(
        model_name: str = 'convnext_tiny',
        pretrained: bool = True,
        num_classes: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> nn.Module:
        """
        Load a ConvNext model.
        
        Args:
            model_name: Name of the ConvNext model variant
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            device: Device to load the model on
            
        Returns:
            Loaded ConvNext model
        """
        if model_name not in ConvNextModelLoader.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not available. "
                f"Choose from: {list(ConvNextModelLoader.AVAILABLE_MODELS.keys())}"
            )
        
        model_fn = ConvNextModelLoader.AVAILABLE_MODELS[model_name]
        
        if pretrained:
            weights = 'IMAGENET1K_V1'
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        
        # Modify classifier if num_classes is different from 1000
        if num_classes != 1000:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        
        model = model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """
        Get information about the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': type(model).__name__
        }


class ModelWrapper(nn.Module):
    """
    Wrapper class for models to facilitate adversarial attacks.
    Handles preprocessing and normalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize ModelWrapper.
        
        Args:
            model: Base model to wrap
            mean: Mean values for normalization (ImageNet default)
            std: Standard deviation values for normalization (ImageNet default)
        """
        super(ModelWrapper, self).__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with normalization.
        
        Args:
            x: Input tensor (assumed to be in [0, 1] range)
            
        Returns:
            Model output
        """
        # Normalize input
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predicted class indices, prediction probabilities)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


def create_convnext_model(
    model_name: str = 'convnext_tiny',
    pretrained: bool = True,
    num_classes: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    wrap_model: bool = True
) -> nn.Module:
    """
    Convenience function to create a ConvNext model.
    
    Args:
        model_name: Name of the ConvNext model variant
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        device: Device to load the model on
        wrap_model: Whether to wrap the model with ModelWrapper
        
    Returns:
        ConvNext model (wrapped or unwrapped)
    """
    loader = ConvNextModelLoader()
    model = loader.load_model(model_name, pretrained, num_classes, device)
    
    if wrap_model:
        model = ModelWrapper(model)
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Loading ConvNext Tiny model...")
    model = create_convnext_model(model_name='convnext_tiny', pretrained=True)
    
    loader = ConvNextModelLoader()
    info = loader.get_model_info(model)
    
    print(f"\nModel Info:")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Model Type: {info['model_type']}")
    
    # Test forward pass
    dummy_input = torch.rand(1, 3, 224, 224)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print("Model loaded successfully!")
