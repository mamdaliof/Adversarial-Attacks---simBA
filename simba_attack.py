"""
SimBA (Simple Black-box Adversarial) Attack Implementation

SimBA is a simple and efficient black-box adversarial attack that only requires
model predictions (no gradients). It uses random direction search to find adversarial
perturbations.

Reference: "Simple Black-box Adversarial Attacks" (Guo et al., 2019)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable
from tqdm import tqdm
import time


class SimBAAttack:
    """
    SimBA (Simple Black-box Adversarial) Attack.
    
    This attack perturbs images by randomly selecting pixels/directions
    and adjusting them to fool the classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        max_iterations: int = 10000,
        pixel_attack: bool = True,
        targeted: bool = False,
        frequency_domain: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SimBA attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum L-infinity perturbation magnitude
            max_iterations: Maximum number of iterations
            pixel_attack: If True, use pixel-wise attack; otherwise use DCT
            targeted: If True, perform targeted attack
            frequency_domain: If True, attack in frequency domain (DCT)
            device: Device to run attack on
        """
        self.model = model.to(device)
        self.model.eval()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.pixel_attack = pixel_attack
        self.targeted = targeted
        self.frequency_domain = frequency_domain
        self.device = device
        
    def _get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def _get_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class."""
        probs = self._get_probs(x)
        return torch.argmax(probs, dim=1)
    
    def _is_adversarial(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_target: Optional[torch.Tensor] = None
    ) -> bool:
        """Check if the example is adversarial."""
        pred = self._get_pred(x)
        
        if self.targeted:
            return pred == y_target
        else:
            return pred != y_true
    
    def _expand_vector(
        self,
        x: torch.Tensor,
        basis: torch.Tensor
    ) -> torch.Tensor:
        """Expand a basis vector to the full perturbation."""
        return basis
    
    def _basis_vector(
        self,
        shape: Tuple[int, ...],
        index: int
    ) -> torch.Tensor:
        """Generate a basis vector (one-hot)."""
        basis = torch.zeros(shape, device=self.device)
        basis.view(-1)[index] = 1
        return basis
    
    def _dct_basis_vector(
        self,
        shape: Tuple[int, ...],
        index: int
    ) -> torch.Tensor:
        """Generate a DCT basis vector."""
        # Simplified DCT basis generation
        # In practice, you would use proper DCT basis functions
        basis = torch.zeros(shape, device=self.device)
        basis.view(-1)[index] = 1
        return basis
    
    def attack_untargeted(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, bool, int, float]:
        """
        Perform untargeted SimBA attack.
        
        Args:
            x: Input image (batch_size, channels, height, width) in [0, 1]
            y_true: True label
            verbose: Whether to print progress
            
        Returns:
            Tuple of (adversarial example, success, num_queries, perturbation_norm)
        """
        x_adv = x.clone()
        n_queries = 0
        
        # Get original prediction
        if self._is_adversarial(x_adv, y_true):
            return x_adv, True, n_queries, 0.0
        
        # Get dimensions
        batch_size, channels, height, width = x.shape
        n_dims = channels * height * width
        
        # Random order of dimensions to perturb
        perm = torch.randperm(n_dims)
        
        if verbose:
            pbar = tqdm(total=self.max_iterations, desc="SimBA Attack")
        
        for i in range(self.max_iterations):
            # Select dimension to perturb
            dim_idx = perm[i % n_dims].item()
            
            # Generate basis vector
            if self.frequency_domain:
                delta = self._dct_basis_vector((channels, height, width), dim_idx)
            else:
                delta = self._basis_vector((channels, height, width), dim_idx)
            
            # Try positive perturbation
            x_trial_pos = torch.clamp(x_adv + self.epsilon * delta, 0, 1)
            n_queries += 1
            
            if self._is_adversarial(x_trial_pos, y_true):
                x_adv = x_trial_pos
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({'queries': n_queries})
                break
            
            # Get probabilities for comparison
            probs_orig = self._get_probs(x_adv)
            probs_pos = self._get_probs(x_trial_pos)
            
            # Check if positive perturbation reduces true class probability
            if probs_pos[0, y_true] < probs_orig[0, y_true]:
                x_adv = x_trial_pos
            else:
                # Try negative perturbation
                x_trial_neg = torch.clamp(x_adv - self.epsilon * delta, 0, 1)
                n_queries += 1
                
                if self._is_adversarial(x_trial_neg, y_true):
                    x_adv = x_trial_neg
                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix({'queries': n_queries})
                    break
                
                probs_neg = self._get_probs(x_trial_neg)
                
                if probs_neg[0, y_true] < probs_orig[0, y_true]:
                    x_adv = x_trial_neg
            
            if verbose and i % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({'queries': n_queries})
        
        if verbose:
            pbar.close()
        
        success = self._is_adversarial(x_adv, y_true)
        perturbation = torch.norm(x_adv - x).item()
        
        return x_adv, success, n_queries, perturbation
    
    def attack_targeted(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_target: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, bool, int, float]:
        """
        Perform targeted SimBA attack.
        
        Args:
            x: Input image in [0, 1]
            y_true: True label
            y_target: Target label
            verbose: Whether to print progress
            
        Returns:
            Tuple of (adversarial example, success, num_queries, perturbation_norm)
        """
        x_adv = x.clone()
        n_queries = 0
        
        # Check if already misclassified as target
        if self._is_adversarial(x_adv, y_true, y_target):
            return x_adv, True, n_queries, 0.0
        
        # Get dimensions
        batch_size, channels, height, width = x.shape
        n_dims = channels * height * width
        
        # Random order of dimensions to perturb
        perm = torch.randperm(n_dims)
        
        if verbose:
            pbar = tqdm(total=self.max_iterations, desc="SimBA Targeted Attack")
        
        for i in range(self.max_iterations):
            # Select dimension to perturb
            dim_idx = perm[i % n_dims].item()
            
            # Generate basis vector
            if self.frequency_domain:
                delta = self._dct_basis_vector((channels, height, width), dim_idx)
            else:
                delta = self._basis_vector((channels, height, width), dim_idx)
            
            # Try positive perturbation
            x_trial_pos = torch.clamp(x_adv + self.epsilon * delta, 0, 1)
            n_queries += 1
            
            if self._is_adversarial(x_trial_pos, y_true, y_target):
                x_adv = x_trial_pos
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({'queries': n_queries})
                break
            
            # Get probabilities
            probs_orig = self._get_probs(x_adv)
            probs_pos = self._get_probs(x_trial_pos)
            
            # Check if positive perturbation increases target class probability
            if probs_pos[0, y_target] > probs_orig[0, y_target]:
                x_adv = x_trial_pos
            else:
                # Try negative perturbation
                x_trial_neg = torch.clamp(x_adv - self.epsilon * delta, 0, 1)
                n_queries += 1
                
                if self._is_adversarial(x_trial_neg, y_true, y_target):
                    x_adv = x_trial_neg
                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix({'queries': n_queries})
                    break
                
                probs_neg = self._get_probs(x_trial_neg)
                
                if probs_neg[0, y_target] > probs_orig[0, y_target]:
                    x_adv = x_trial_neg
            
            if verbose and i % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({'queries': n_queries})
        
        if verbose:
            pbar.close()
        
        success = self._is_adversarial(x_adv, y_true, y_target)
        perturbation = torch.norm(x_adv - x).item()
        
        return x_adv, success, n_queries, perturbation
    
    def attack(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, bool, int, float]:
        """
        Perform SimBA attack (targeted or untargeted).
        
        Args:
            x: Input image in [0, 1]
            y_true: True label
            y_target: Target label (for targeted attack)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (adversarial example, success, num_queries, perturbation_norm)
        """
        if self.targeted and y_target is None:
            raise ValueError("Target label must be provided for targeted attack")
        
        if self.targeted:
            return self.attack_targeted(x, y_true, y_target, verbose)
        else:
            return self.attack_untargeted(x, y_true, verbose)
    
    def evaluate_attack_success_rate(
        self,
        data_loader,
        max_samples: Optional[int] = None
    ) -> dict:
        """
        Evaluate attack success rate on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with attack statistics
        """
        total = 0
        successful = 0
        total_queries = 0
        total_perturbation = 0.0
        
        print("Evaluating attack success rate...")
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_samples and total >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if max_samples and total >= max_samples:
                    break
                
                x = images[i:i+1]
                y = labels[i:i+1]
                
                # Perform attack
                x_adv, success, n_queries, perturbation = self.attack(
                    x, y, verbose=False
                )
                
                total += 1
                if success:
                    successful += 1
                    total_queries += n_queries
                    total_perturbation += perturbation
                
                if total % 10 == 0:
                    print(f"Processed {total} samples, Success rate: {successful/total*100:.2f}%")
        
        avg_queries = total_queries / successful if successful > 0 else 0
        avg_perturbation = total_perturbation / successful if successful > 0 else 0
        
        results = {
            'total_samples': total,
            'successful_attacks': successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_queries': avg_queries,
            'avg_perturbation': avg_perturbation
        }
        
        return results


if __name__ == "__main__":
    from models import create_convnext_model
    
    print("SimBA Attack module loaded successfully!")
    print("\nExample usage:")
    print("model = create_convnext_model('convnext_tiny', pretrained=True)")
    print("attack = SimBAAttack(model, epsilon=0.2, max_iterations=10000)")
    print("x_adv, success, queries, pert = attack.attack(x, y_true)")
