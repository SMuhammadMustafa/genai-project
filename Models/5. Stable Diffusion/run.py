import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLTrainer, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model, save_pretrained_peft_model
import torch

# Configurations
model_name = "stabilityai/stable-diffusion-xl-base-1.0"  # Base SDXL model
dataset_path = "./custom_dataset"  # Path to your dataset
output_dir = "./sdxl-lora-finetuned"  # Directory to save LoRA weights
batch_size = 4
num_epochs = 5
learning_rate = 1e-4
lora_r = 4  # Rank of LoRA
lora_alpha = 32  # Alpha scaling factor
lora_dropout = 0.1  # Dropout rate for LoRA layers

# 1. Load Dataset
def preprocess_dataset(example):
    """Preprocess dataset entries."""
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    example["input_ids"] = tokenizer(
        example["caption"], truncation=True, padding="max_length", max_length=77
    )["input_ids"]
    return example

dataset = load_dataset("imagefolder", data_dir=dataset_path)
dataset = dataset.map(preprocess_dataset)
train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

# 2. Load Stable Diffusion XL Model
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipeline.to("cuda")

# Use LoRA for the UNet
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float16)
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["to_q", "to_v"],  # LoRA applied to specific modules
    lora_dropout=lora_dropout,
    bias="none",
    task_type="UNet"
)
lora_unet = get_peft_model(unet, lora_config)

# 3. Define Optimizer
optimizer = torch.optim.AdamW(lora_unet.parameters(), lr=learning_rate)

# 4. Fine-tune LoRA Parameters
lora_unet.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_unet.to(device)

for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward pass
        images = batch["image"].to(device, dtype=torch.float16)
        captions = batch["input_ids"].to(device)
        outputs = lora_unet(images, captions)

        # Compute loss
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {loss.item()}")

# 5. Save LoRA Weights as `safetensors`
save_pretrained_peft_model(lora_unet, output_dir, safe_serialization=True)
print(f"LoRA weights saved to {output_dir} in safetensors format.")
