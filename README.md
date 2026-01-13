# Adversarial Attacks on ConvNext Models using SimBA

This project implements adversarial attacks (specifically SimBA - Simple Black-box Adversarial attacks) on ConvNext models, along with various defense mechanisms.

## 📋 Overview

This repository provides a complete pipeline for:
- Loading and using ConvNext models (Tiny, Small, Base, Large)
- Fine-tuning models with warm-up schedulers and advanced training features
- Performing SimBA adversarial attacks (black-box attacks requiring only model predictions)
- Implementing defense mechanisms against adversarial attacks
- Evaluating attack success rates and defense effectiveness

## 🏗️ Project Structure

```
.
├── models.py           # ConvNext model definitions and loading utilities
├── fine_tune.py        # Fine-tuning with warm-up scheduler and training utilities
├── simba_attack.py     # SimBA adversarial attack implementation
├── defense.py          # Defense mechanisms (input transformations, ensemble, detection)
├── main.py             # Main orchestration script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster execution)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mamdaliof/Adversarial-Attacks---simBA.git
cd Adversarial-Attacks---simBA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

Run the complete pipeline with default settings:
```bash
python main.py --model convnext_tiny --dataset cifar10 --attack --defense input_transform
```

### Advanced Options

#### 1. Model Selection
Choose from different ConvNext variants:
```bash
python main.py --model convnext_small  # Options: convnext_tiny, convnext_small, convnext_base, convnext_large
```

#### 2. Fine-tuning
Fine-tune the model before attacking:
```bash
python main.py --fine-tune --epochs 20 --learning-rate 1e-4 --warmup-epochs 5
```

#### 3. Attack Configuration
Configure SimBA attack parameters:
```bash
python main.py --attack --epsilon 0.2 --max-iterations 10000 --max-samples 100
```

For targeted attacks:
```bash
python main.py --attack --targeted --epsilon 0.2
```

#### 4. Defense Mechanisms
Choose from different defense strategies:
```bash
# Input transformation defense
python main.py --defense input_transform

# Ensemble defense
python main.py --defense ensemble

# Adversarial detector
python main.py --defense detector

# No defense
python main.py --defense none
```

### Complete Example

```bash
python main.py \
    --model convnext_tiny \
    --dataset cifar10 \
    --data-path ./data \
    --fine-tune \
    --epochs 10 \
    --learning-rate 1e-4 \
    --warmup-epochs 2 \
    --attack \
    --epsilon 0.2 \
    --max-iterations 5000 \
    --max-samples 50 \
    --defense input_transform \
    --output-dir ./results \
    --device cuda
```

## 📚 Module Details

### 1. `models.py`
- **ConvNextModelLoader**: Load pre-trained ConvNext models
- **ModelWrapper**: Wrapper for handling preprocessing and normalization
- **create_convnext_model()**: Convenience function to create models

Example:
```python
from models import create_convnext_model

model = create_convnext_model(
    model_name='convnext_tiny',
    pretrained=True,
    num_classes=1000
)
```

### 2. `fine_tune.py`
- **WarmupCosineScheduler**: Learning rate scheduler with warm-up and cosine annealing
- **EarlyStopping**: Early stopping based on validation loss
- **ModelTrainer**: Complete training pipeline with mixed precision support

Features:
- Learning rate warm-up
- Cosine annealing scheduler
- Early stopping
- Model checkpointing
- Mixed precision training (automatic on CUDA)

Example:
```python
from fine_tune import ModelTrainer

trainer = ModelTrainer(model, device='cuda')
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=1e-4,
    warmup_epochs=5
)
```

### 3. `simba_attack.py`
- **SimBAAttack**: Implementation of Simple Black-box Adversarial attack

Features:
- Untargeted and targeted attacks
- Pixel-wise and frequency domain attacks
- Query-efficient black-box attack
- Attack success rate evaluation

Example:
```python
from simba_attack import SimBAAttack

attack = SimBAAttack(
    model=model,
    epsilon=0.2,
    max_iterations=10000,
    targeted=False
)

x_adv, success, queries, perturbation = attack.attack(x, y_true)
```

### 4. `defense.py`
- **InputTransformationDefense**: Defense using input transformations
  - JPEG compression
  - Bit depth reduction
  - Gaussian blur
  - Total variation denoising

- **EnsembleDefense**: Ensemble of multiple models
  - Soft voting (probability averaging)
  - Hard voting (majority voting)

- **AdversarialDetector**: Detect adversarial examples
  - Confidence-based detection
  - Reconstruction-based detection

- **AdversarialTrainingDefense**: Train with adversarial examples

Example:
```python
from defense import InputTransformationDefense

defense = InputTransformationDefense(
    model=model,
    transforms_list=['jpeg_compression', 'gaussian_blur']
)

predictions, probabilities = defense.predict(x)
```

### 5. `main.py`
Orchestrates the complete pipeline:
1. Load ConvNext model
2. Fine-tune (optional)
3. Perform SimBA attack
4. Apply defense mechanisms
5. Evaluate and save results

## 🔬 SimBA Attack

SimBA (Simple Black-box Adversarial) is an efficient black-box attack that:
- Requires only model predictions (no gradients)
- Uses random direction search
- Modifies pixels iteratively to fool the classifier
- Is query-efficient compared to other black-box methods

**Reference**: "Simple Black-box Adversarial Attacks" (Guo et al., 2019)

## 🛡️ Defense Mechanisms

### Input Transformation
Applies transformations to remove adversarial perturbations:
- JPEG compression
- Bit depth reduction
- Gaussian blur
- Total variation minimization

### Ensemble Defense
Uses multiple models to make robust predictions through voting.

### Adversarial Detection
Identifies adversarial examples based on:
- Prediction confidence
- Reconstruction error

### Adversarial Training
Trains the model on both clean and adversarial examples.

## 📊 Results

Results are saved in the output directory (default: `./results/`) with:
- `config.json`: Configuration used
- `attack_results.json`: Attack success rate and statistics
- `training_history.json`: Training metrics (if fine-tuning was performed)
- `checkpoints/`: Model checkpoints

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- ConvNext models from torchvision
- SimBA attack based on "Simple Black-box Adversarial Attacks" by Guo et al.
- PyTorch framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.