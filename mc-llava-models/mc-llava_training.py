#mc-llava-training
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# --- 1. Configuration ---
model_id = "visheratin/MC-LLaVA-3b"
revision = "06bc212f5ae47e362b98afe3abd929eb603ce9ba"
train_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_cache/train"
val_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_cache/val"
save_path = "/home/jme138/model_attempts/llava_fine_tuned_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load model and processor ---
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revision,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.to(device)

# --- 3. Dataset from .pt Cache Files ---
class CachedDataset(Dataset):
    def __init__(self, directory):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        sequences = [item[key] for item in batch]

        if key in ['input_ids', 'labels']:
            max_len = max(seq.size(0) for seq in sequences)
            dtype = sequences[0].dtype
            device = sequences[0].device
            pad_val = -100 if key == 'labels' else 0

            padded = []
            for seq in sequences:
                pad_len = max_len - seq.size(0)
                padding = torch.full((pad_len,), pad_val, dtype=dtype, device=device)
                padded.append(torch.cat([seq, padding]))

            collated[key] = torch.stack(padded)
        else:
            collated[key] = torch.stack(sequences)

    return collated

# --- 4. DataLoader Setup ---
train_loader = DataLoader(CachedDataset(train_dir), batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(CachedDataset(val_dir), batch_size=2, shuffle=False, collate_fn=collate_fn)

# --- 5. Training Loop ---
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Avg Train Loss: {avg_train_loss:.4f}")

    # --- 6. Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Avg Val Loss: {avg_val_loss:.4f}")

# --- 7. Save Fine-Tuned Model ---
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print(f"\nModel saved to {save_path}")
