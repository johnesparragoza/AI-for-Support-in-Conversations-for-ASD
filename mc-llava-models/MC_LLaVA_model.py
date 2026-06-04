#mc-llava-model

import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig




csv_path = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/results.csv"
image_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/images_"

df = pd.read_csv(csv_path, sep='|')
df.columns = [col.strip() for col in df.columns]
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(image_dir, x))

# Rename for clarity
df_final = df[['image_path', 'comment']].rename(columns={'comment': 'caption'})

# Only keep samples with valid image files
df_final = df_final[df_final['image_path'].apply(os.path.exists)].reset_index(drop=True)

"""
Loads image and caption per sample.

Prepares it with a processor (like LlavaProcessor.from_pretrained(...)).

Adds labels for supervised fine-tuning.
"""


class Flickr30kDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe
        self.processor = processor
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        caption = row['caption']

        # Construct the prompt
        prompt = f"<image>\nDescribe this image using short and simple sentences.\nCaption: {caption}\nNarrative:"

        # IMPORTANT: Wrap `image` in a list for compatibility with LLaVA processor
        inputs = self.processor(prompt, images=[image], model=self.model, return_tensors="pt")

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Labels are just the input_ids cloned
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# make each sequence the same length (using padding) 
def collate_fn(batch):
    # batch is a list of dicts returned by your Dataset's __getitem__
    # For example, each item in batch might be {'input_ids': ..., 'labels': ..., ...}
    # You need to pad all sequences to the same length
    input_keys = batch[0].keys()
    collated = {}
    for key in input_keys:
        # Use processor's pad method if available, otherwise pad manually
        if key in ['input_ids', 'labels']:
            # Collect all sequences
            sequences = [item[key] for item in batch]
            # Pad to the max length in this batch
            max_len = max(seq.shape[0] for seq in sequences)
            padded = [torch.cat([seq, torch.full((max_len - seq.shape[0],), fill_value=-100 if key=='labels' else 0, dtype=seq.dtype)]) for seq in sequences]
            collated[key] = torch.stack(padded)
        else:
            # Stack as usual
            collated[key] = torch.stack([item[key] for item in batch])
    return collated

train_loader = DataLoader(
    dataset,
    batch_size=YOUR_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn  # Use your custom collate function
)


model_id = "visheratin/MC-LLaVA-3b"
revision = "06bc212f5ae47e362b98afe3abd929eb603ce9ba"  # You can replace this with   another commit if needed


# Load processor and model with trust_remote_code enabled
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically picks the best device (GPU if available)
    trust_remote_code=True,
    revision=revision
)

# Dataset + Dataloader
dataset = Flickr30kDataset(df_final, processor)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True) #BATCH SIZE CHANGE

#training of model
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 5  # Increase once stable

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    num_batches = 0

    for step, batch in enumerate(tqdm(train_loader)):
        for k in batch:
            batch[k] = batch[k].to(model.device)
        
        model_inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = model(**model_inputs)
        logits = outputs['logits']
        batch_size, seq_len, vocab_size = logits.shape
        labels = batch['labels']

        # Adjust label shape if needed
        if labels.shape[1] != seq_len:
            if labels.shape[1] < seq_len:
                padding = torch.full((batch_size, seq_len - labels.shape[1]), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, padding], dim=1)
            else:
                labels = labels[:, :seq_len]

        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

        # --- Accuracy calculation ---
        # Get predictions: [batch_size, seq_len]
        preds = logits.argmax(dim=-1)
        # Mask out padding in labels
        mask = (labels != -100)
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        epoch_correct += correct
        epoch_total += total

    avg_loss = epoch_loss / num_batches
    accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4%}")

    
