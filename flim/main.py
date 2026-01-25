import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as plt
from simba import SimBA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load the Model & Processor
checkpoint = "facebook/convnextv2-tiny-1k-224"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(checkpoint).to(device)
model.eval()

# load dataset
dataset = load_dataset("mrm8488/ImageNet1K-val", split="train")

#Preprocessing Function
def transforms(examples):
    inputs = image_processor([img.convert("RGB") for img in examples["image"]],return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

# Apply transforms to the dataset
prepared_ds = dataset.with_transform(transforms)

# Now we have prepared_ds, this is the dataset ready to use a SimBA attack on.
dataloader = DataLoader(prepared_ds, batch_size=64, shuffle=True)

# Check a single image
sample = prepared_ds[1000]
print("--- Dataset Item Check ---")
print(f"Keys available: {sample.keys()}")
print(f"Pixel values shape: {sample['pixel_values'].shape}")
print(f"Label: {sample['labels']}")


#############################
#### Initializing Attack ####
#############################
simba_attack = SimBA(model=model, dataset=prepared_ds, image_size=224)

total_images = 10
correct_after_attack = 0

print(f"Starting SimBA attack on {total_images} images...")

#The Attack Loop, iterate manually through the dataset to handle single images
for i in tqdm(range(total_images)):
    sample = prepared_ds[i]
    
    image = sample['pixel_values'].to(device)  # Shape: [3, 224, 224]
    label = torch.tensor([sample['labels']]).to(device)
    
    with torch.no_grad():
        clean_pred = simba_attack.get_preds(image.unsqueeze(0))
    
    if clean_pred != label:
        continue
        
    adv_image = simba_attack.simba_single(image, label, num_iters=1000, epsilon=0.2)
    
    with torch.no_grad():
        final_pred = simba_attack.get_preds(adv_image.unsqueeze(0))
        
    if final_pred == label:
        correct_after_attack += 1

adv_accuracy = (correct_after_attack / total_images) * 100
print(f"\n--- Attack Results ---")
print(f"Adversarial Accuracy: {adv_accuracy:.2f}%")
print(f"Attack Success Rate: {100 - adv_accuracy:.2f}%")