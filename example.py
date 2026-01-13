"""
Example Script: Quick Start with SimBA Attack on ConvNext

This script demonstrates a simple example of using the SimBA attack
on a ConvNext model with a few sample images.
"""

import torch
from models import create_convnext_model
from simba_attack import SimBAAttack
from defense import InputTransformationDefense
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def create_sample_images(num_samples=5):
    """Create random sample images for demonstration."""
    print(f"Creating {num_samples} random sample images...")
    images = torch.rand(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))
    return images, labels


def demonstrate_attack():
    """Demonstrate a simple adversarial attack."""
    print("=" * 80)
    print("SimBA Adversarial Attack Demonstration")
    print("=" * 80)
    
    # 1. Load model
    print("\n[1] Loading ConvNext Tiny model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_convnext_model(
        model_name='convnext_tiny',
        pretrained=True,
        device=device
    )
    print("✓ Model loaded successfully")
    
    # 2. Create sample data
    print("\n[2] Creating sample data...")
    images, labels = create_sample_images(num_samples=3)
    images = images.to(device)
    labels = labels.to(device)
    print(f"✓ Created {len(images)} sample images")
    
    # 3. Get clean predictions
    print("\n[3] Getting clean predictions...")
    with torch.no_grad():
        clean_outputs = model(images)
        clean_preds = torch.argmax(clean_outputs, dim=1)
    
    print("Clean predictions:")
    for i in range(len(images)):
        print(f"  Image {i+1}: True label={labels[i].item()}, "
              f"Predicted={clean_preds[i].item()}, "
              f"Match={clean_preds[i] == labels[i]}")
    
    # 4. Perform attack
    print("\n[4] Performing SimBA attack...")
    attack = SimBAAttack(
        model=model,
        epsilon=0.2,
        max_iterations=1000,
        targeted=False,
        device=device
    )
    
    successful_attacks = 0
    total_queries = 0
    
    for i in range(len(images)):
        print(f"\n  Attacking image {i+1}...")
        x = images[i:i+1]
        y = labels[i:i+1]
        
        x_adv, success, queries, perturbation = attack.attack(
            x, y, verbose=False
        )
        
        if success:
            successful_attacks += 1
            total_queries += queries
            
        # Get adversarial prediction
        with torch.no_grad():
            adv_output = model(x_adv)
            adv_pred = torch.argmax(adv_output, dim=1)
        
        print(f"  - Success: {success}")
        print(f"  - Queries: {queries}")
        print(f"  - Perturbation: {perturbation:.4f}")
        print(f"  - Original prediction: {clean_preds[i].item()}")
        print(f"  - Adversarial prediction: {adv_pred.item()}")
    
    print(f"\n✓ Attack completed: {successful_attacks}/{len(images)} successful")
    if successful_attacks > 0:
        print(f"  Average queries: {total_queries / successful_attacks:.2f}")
    
    # 5. Apply defense
    print("\n[5] Testing defense mechanism...")
    defense = InputTransformationDefense(
        model=model,
        transforms_list=['gaussian_blur', 'bit_depth_reduction'],
        device=device
    )
    print("✓ Defense initialized")
    
    # Test defense on one adversarial example
    if successful_attacks > 0:
        x_adv_sample = images[0:1]
        
        # Create an adversarial example for testing
        x_adv_test, _, _, _ = attack.attack(
            x_adv_sample, labels[0:1], verbose=False
        )
        
        # Predict with defense
        defended_pred, defended_prob = defense.predict(x_adv_test)
        
        print(f"\nDefense results on adversarial example:")
        print(f"  - Original label: {labels[0].item()}")
        print(f"  - Defended prediction: {defended_pred.item()}")
        print(f"  - Confidence: {defended_prob.max().item():.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Try with real images using datasets like CIFAR-10 or ImageNet")
    print("2. Experiment with different epsilon values and attack parameters")
    print("3. Compare different defense mechanisms")
    print("4. Fine-tune models on your custom dataset")
    print("\nRun 'python main.py --help' for more options")


if __name__ == "__main__":
    demonstrate_attack()
