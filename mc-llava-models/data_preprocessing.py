import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Config ---
csv_path = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/results.csv"
image_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/images_"
model_id = "visheratin/MC-LLaVA-3b"
revision = "06bc212f5ae47e362b98afe3abd929eb603ce9ba"
cache_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_cache"
os.makedirs(cache_dir, exist_ok=True)

# --- Load Caption CSV ---
df = pd.read_csv(csv_path, sep='|')
df.columns = [col.strip() for col in df.columns]
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(image_dir, x))
df = df.rename(columns={'comment': 'caption'})
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

# --- Load processor and model ---
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)

# --- Preprocess & Cache ---
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing + Caching"):
    try:
        image = Image.open(row['image_path']).convert("RGB")
        caption = row['caption']
        prompt = f"<image>\nDescribe this image using short and simple sentences.\nCaption: {caption}\nNarrative:"
        inputs = processor(prompt, images=[image], model=model, return_tensors="pt", padding="longest")
        inputs["labels"] = inputs["input_ids"].clone()
        torch.save(inputs, os.path.join(cache_dir, f"sample_{idx}.pt"))
    except Exception as e:
        print(f"[{idx}] Error: {e}")
