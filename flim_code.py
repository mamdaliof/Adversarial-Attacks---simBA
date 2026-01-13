import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

# Apply transforms to the dataset
prepared_ds = dataset.with_transform(transforms)

#Create DataLoader
def collate_fn(original_list):
    batch = {}
    batch["pixel_values"] = torch.stack([x["pixel_values"] for x in original_list]).squeeze().to(device) # Result is a 4D array (Batch, Channels, Height, Width)
    batch["labels"] = torch.tensor([x["labels"] for x in original_list]).to(device)
    
    return batch

eval_dataloader = DataLoader(prepared_ds, batch_size=32, collate_fn=collate_fn)

# 6. Evaluation Loop
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        outputs = model(batch["pixel_values"])
        predictions = outputs.logits.argmax(-1)
        
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")