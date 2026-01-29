import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as plt
from simba import SimBA
import csv
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#Load the Model & Processor
checkpoint = "facebook/convnextv2-tiny-1k-224"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_normalize=False)
model = AutoModelForImageClassification.from_pretrained(checkpoint).to(device)
model.eval()

# load dataset
dataset = load_dataset("mrm8488/ImageNet1K-val", split="train[0:64]")

#Preprocessing Function
def transforms(examples):
    inputs = image_processor([img.convert("RGB") for img in examples["image"]],return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

# Apply transforms to the dataset
prepared_ds = dataset.with_transform(transforms)

# Now we have prepared_ds, this is the dataset ready to use a SimBA attack on.
dataloader = DataLoader(prepared_ds, batch_size=16, shuffle=False)

# Logging
output_dir = "simba_results"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "attack_log.csv")

with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["img_idx", "true_label", "final_pred", "success", "queries", "l2_norm", "init_p", "final_p"])

#############################
#### Initializing Attack ####
#############################
simba_attack = SimBA(model=model, dataset=prepared_ds, image_size=224)

# total_images = 10
# correct_after_attack = 0

# print(f"Starting SimBA attack on {total_images} images...")

# #The Attack Loop, iterate manually through the dataset to handle single images
# for i in tqdm(range(total_images)):
#     sample = prepared_ds[i]
    
#     image = sample['pixel_values'].to(device)  # Shape: [3, 224, 224]
#     label = torch.tensor([sample['labels']]).to(device)
    
#     with torch.no_grad():
#         clean_pred = simba_attack.get_preds(image.unsqueeze(0))
    
#     if clean_pred != label:
#         continue
        
#     adv_image = simba_attack.simba_single(image, label, num_iters=1000, epsilon=0.2)
    
#     with torch.no_grad():
#         final_pred = simba_attack.get_preds(adv_image.unsqueeze(0))
        
#     if final_pred == label:
#         correct_after_attack += 1

# adv_accuracy = (correct_after_attack / total_images) * 100
# print(f"\n--- Attack Results ---")
# print(f"Adversarial Accuracy: {adv_accuracy:.2f}%")
# print(f"Attack Success Rate: {100 - adv_accuracy:.2f}%")

###################################
#### Running DCT Batch Attack  ####
###################################
# Create lists to store results
img_counter = 0
all_successes = []

print(f"Starting SimBA DCT attack on {len(prepared_ds)} images...")

for batch in tqdm(dataloader):
    images = batch['pixel_values'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        # This returns a tensor of probabilities for the correct class
        initial_probs = simba_attack.get_probs(images, labels).cpu().numpy()

    # Run the batch attack
    adv_images, probs, succs, queries, l2_norms, linf_norms = simba_attack.simba_batch(
        images, 
        labels, 
        max_iters=5000, 
        freq_dims=64, 
        stride=7, 
        epsilon=0.2, 
        order='rand', 
        pixel_attack=False,
        log_every=10
    )
    
    # Store results
    with torch.no_grad():
        final_preds = simba_attack.get_preds(adv_images)
        final_probs = simba_attack.get_probs(adv_images, labels).cpu().numpy()

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for i in range(images.size(0)):

            l2 = torch.norm(adv_images[i] - images[i]).item()
            total_q = queries[i].sum().item()
            is_success = (final_preds[i].item() != labels[i].item())
            
            writer.writerow([
                img_counter, 
                labels[i].item(), 
                final_preds[i].item(), 
                is_success, 
                total_q, 
                l2,
                initial_probs[i],
                final_probs[i]
            ])
            img_counter += 1

# Final Summary
asr = (sum(all_successes) / len(all_successes)) * 100
print(f"\n--- Done! ---")
print(f"Final ASR: {asr:.2f}%")
print(f"Logs saved to: {log_file}")