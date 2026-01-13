"""
Main Script for Adversarial Attacks and Defense on ConvNext Models

This script orchestrates the complete pipeline:
1. Load ConvNext model
2. Optionally fine-tune the model
3. Perform SimBA adversarial attack
4. Apply defense mechanisms
5. Evaluate results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import json
from datetime import datetime

from models import create_convnext_model, ConvNextModelLoader
from fine_tune import ModelTrainer
from simba_attack import SimBAAttack
from defense import (
    InputTransformationDefense,
    EnsembleDefense,
    AdversarialDetector,
    evaluate_defense
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adversarial Attacks and Defense on ConvNext Models'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='convnext_tiny',
        choices=['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
        help='ConvNext model variant'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
        help='Number of output classes'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='Path to dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet',
        choices=['imagenet', 'cifar10', 'cifar100', 'custom'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Fine-tuning arguments
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Perform fine-tuning before attack'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of fine-tuning epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=2,
        help='Number of warmup epochs'
    )
    
    # Attack arguments
    parser.add_argument(
        '--attack',
        action='store_true',
        default=True,
        help='Perform adversarial attack'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.2,
        help='Maximum perturbation magnitude'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10000,
        help='Maximum attack iterations'
    )
    parser.add_argument(
        '--targeted',
        action='store_true',
        help='Perform targeted attack'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum samples to attack'
    )
    
    # Defense arguments
    parser.add_argument(
        '--defense',
        type=str,
        default='input_transform',
        choices=['input_transform', 'ensemble', 'detector', 'none'],
        help='Defense mechanism to use'
    )
    
    # General arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_data_loader(args):
    """
    Create data loader for the specified dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms
    if args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:  # CIFAR-10, CIFAR-100
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    
    # Load dataset
    train_loader = None
    test_loader = None
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_path,
            train=False,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    elif args.dataset == 'imagenet':
        print("Note: For ImageNet, please provide the dataset path.")
        print("Expected structure: data_path/train and data_path/val")
        # For ImageNet, users need to download the dataset separately
        # This is a placeholder
        pass
    
    return train_loader, test_loader


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print("=" * 80)
    print("Adversarial Attacks and Defense on ConvNext Models")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    
    # Step 1: Load Model
    print("\n[Step 1] Loading ConvNext model...")
    model = create_convnext_model(
        model_name=args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        device=args.device,
        wrap_model=True
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    loader = ConvNextModelLoader()
    info = loader.get_model_info(model)
    print(f"Model loaded: {info['model_type']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    
    # Step 2: Fine-tuning (optional)
    if args.fine_tune:
        print("\n[Step 2] Fine-tuning model...")
        train_loader, val_loader = get_data_loader(args)
        
        if train_loader is None:
            print("Warning: No training data available. Skipping fine-tuning.")
        else:
            trainer = ModelTrainer(model, device=args.device)
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                warmup_epochs=args.warmup_epochs,
                save_dir=os.path.join(args.output_dir, 'checkpoints')
            )
            
            # Save training history
            history_path = os.path.join(args.output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Training history saved to {history_path}")
    else:
        print("\n[Step 2] Skipping fine-tuning...")
    
    # Step 3: Perform Attack
    if args.attack:
        print("\n[Step 3] Performing SimBA adversarial attack...")
        
        # Get test data
        _, test_loader = get_data_loader(args)
        
        if test_loader is None:
            print("Warning: No test data available. Creating dummy data for demonstration.")
            # Create dummy data for demonstration
            dummy_images = torch.rand(args.max_samples, 3, 224, 224).to(args.device)
            dummy_labels = torch.randint(0, args.num_classes, (args.max_samples,)).to(args.device)
            test_loader = [(dummy_images[i:i+1], dummy_labels[i:i+1]) 
                          for i in range(args.max_samples)]
        
        # Initialize attack
        attack = SimBAAttack(
            model=model,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            targeted=args.targeted,
            device=args.device
        )
        
        # Evaluate attack
        attack_results = attack.evaluate_attack_success_rate(
            data_loader=test_loader,
            max_samples=args.max_samples
        )
        
        print("\nAttack Results:")
        print(f"Total Samples: {attack_results['total_samples']}")
        print(f"Successful Attacks: {attack_results['successful_attacks']}")
        print(f"Success Rate: {attack_results['success_rate']*100:.2f}%")
        print(f"Average Queries: {attack_results['avg_queries']:.2f}")
        print(f"Average Perturbation: {attack_results['avg_perturbation']:.4f}")
        
        # Save attack results
        results_path = os.path.join(args.output_dir, 'attack_results.json')
        with open(results_path, 'w') as f:
            json.dump(attack_results, f, indent=4)
        print(f"Attack results saved to {results_path}")
    else:
        print("\n[Step 3] Skipping attack...")
    
    # Step 4: Apply Defense
    if args.defense != 'none':
        print(f"\n[Step 4] Applying {args.defense} defense...")
        
        if args.defense == 'input_transform':
            defense = InputTransformationDefense(
                model=model,
                transforms_list=['jpeg_compression', 'bit_depth_reduction', 'gaussian_blur'],
                device=args.device
            )
        elif args.defense == 'ensemble':
            # For ensemble, create multiple instances of the same model
            # In practice, you might want to use different models
            models = [model for _ in range(3)]
            defense = EnsembleDefense(
                models=models,
                device=args.device,
                voting='soft'
            )
        elif args.defense == 'detector':
            defense = AdversarialDetector(
                model=model,
                device=args.device,
                threshold=0.5
            )
        
        print(f"Defense mechanism '{args.defense}' initialized.")
        
        # Note: Evaluation of defense would require running attacks again
        # This is left as an exercise to avoid excessive computation
        print("Note: To fully evaluate defense, re-run attack with defense enabled.")
    else:
        print("\n[Step 4] No defense applied...")
    
    # Step 5: Summary
    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Fine-tuning: {'Yes' if args.fine_tune else 'No'}")
    print(f"Attack: {'Yes' if args.attack else 'No'}")
    print(f"Defense: {args.defense}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    print("\nDone!")


if __name__ == "__main__":
    main()
