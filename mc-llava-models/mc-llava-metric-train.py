import os, csv, io, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoProcessor, AutoModel, AutoTokenizer,
    CLIPProcessor, CLIPModel
)
from peft import LoraConfig, get_peft_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pycocoevalcap.cider.cider import Cider

# ================================
# Configuration and Setup
# ================================
captions_file = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/results.csv"
images_dir = "/home/jme138/model_attempts/30k_dataset/flickr30k_images/images_"
model_name = "visheratin/MC-LLaVA-3b"  # Vision-language model
batch_size, accumulation_steps = 4, 2
resume = False
checkpoint_dir = "checkpoints_mc_llava"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
csv_log_path = os.path.join(checkpoint_dir, "training_metrics.csv")
max_to_keep = 3  # Number of checkpoints to retain
prompt_prefix = "Generate a simple story about the image:\n<image>\n"  # Instructional prompt for narrative generation

# ================================
# Load Dataset (Image-Caption Pairs)
# ================================
df = pd.read_csv(captions_file, header=None, names=["image_name","idx","caption"], sep="\\|", engine="python")
df["image_name"] = df["image_name"].str.strip()
df["image_path"] = df["image_name"].apply(lambda x: os.path.join(images_dir, x))
df["caption"] = df["caption"].astype(str).str.strip()
df = df[df["image_path"].apply(os.path.exists) & df["caption"].str.len().gt(0)]
print(f"Loaded {len(df)} valid pairs.")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ================================
# Load Model, Tokenizer, and Processor
# ================================
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"  # Optional: Automatically use GPU if available
)

# ================================
# LoRA: Low-Rank Adaptation for Parameter-Efficient Fine-Tuning
# ================================
# Instead of updating all weights, LoRA injects trainable matrices into certain layers (e.g., q_proj, v_proj)
# This reduces memory use and accelerates training for large models
lora_cfg = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=32, lora_dropout=0.05)
model = get_peft_model(model, lora_cfg)

# ================================
# Custom Dataset Class for Flickr30K
# ================================
class Flickr30KDataset(Dataset):
    def __init__(self, df, proc):
        self.df, self.proc = df.reset_index(drop=True), proc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert("RGB")
        # Prepend instructional prompt for narrative generation
        caption = f"{prompt_prefix}{row.caption}"
        enc = self.proc(text=caption, images=[img], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        enc = {k: v.squeeze(0) for k,v in enc.items()}
        enc["labels"] = enc["input_ids"]  # Set labels for teacher forcing loss
        return enc

# ================================
# Collate Function for DataLoader
# ================================
def collate_fn(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}

# Data Loaders
train_loader = DataLoader(Flickr30KDataset(train_df, processor), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Flickr30KDataset(val_df, processor), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ================================
# Setup for Training
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Metrics (CIDEr and CLIPScore)
loss_vals, cider_vals, clip_vals = [], [], []
cider_scorer = Cider()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
    trust_remote_code=False,
    use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Checkpointing
best_cider, start_epoch = 0.0, 0
with open(csv_log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "loss", "cider", "clipscore"])

# Resume from checkpoint if enabled
if resume:
    epochs = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pt")])
    if epochs:
        ckpt = torch.load(os.path.join(checkpoint_dir, epochs[-1]))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_cider = ckpt.get("cider", 0.0)
        print(f"Resumed from epoch {start_epoch}")

# ================================
# Main Training Loop
# ================================
for epoch in range(start_epoch, 5):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        if (step + 1) % accumulation_steps == 0 or step+1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps

    avg_loss = total_loss / len(train_loader)
    loss_vals.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # Save and manage checkpoints
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss, "cider": best_cider}, ckpt_path)
    ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")])
    for old in ckpts[:-max_to_keep]:
        os.remove(os.path.join(checkpoint_dir, old))

    # ================================
    # Validation Step
    # ================================
    model.eval()
    all_preds, all_refs, all_imgs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            pixel_values = batch["pixel_values"].to(device)
            # Use same prompt prefix during generation
            prompts = [prompt_prefix] * pixel_values.size(0)
            inputs = processor(text=prompts, images=pixel_values, return_tensors="pt", padding=True).to(device)
            gen = model.generate(**inputs, max_new_tokens=32, num_beams=5)
            all_preds += processor.batch_decode(gen, skip_special_tokens=True)
            all_refs += processor.batch_decode(batch["labels"], skip_special_tokens=True)
            all_imgs += pixel_values.cpu()

    # CIDEr: Consensus-based image caption quality metric
    gts = {i: [ref] for i, ref in enumerate(all_refs)}
    res = {i: [pred] for i, pred in enumerate(all_preds)}
    cider, _ = cider_scorer.compute_score(gts, res)
    cider_vals.append(cider)

    # CLIPScore: Cosine similarity between image and caption embeddings
    clip_scores = []
    for img_t, cap in zip(all_imgs, all_preds):
        img_t = img_t.clamp(0,1)
        pil = transforms.ToPILImage()(img_t)
        inp = clip_processor(images=pil, text=cap, return_tensors="pt").to(device)
        out = clip_model(**inp)
        clip_scores.append(torch.cosine_similarity(out.image_embeds, out.text_embeds).item())
    avg_clip = sum(clip_scores)/len(clip_scores)
    clip_vals.append(avg_clip)
    print(f"[Epoch {epoch+1}] CIDEr: {cider:.4f} | CLIPScore: {avg_clip:.4f}")

    # Log metrics to CSV
    with open(csv_log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, avg_loss, cider, avg_clip])

    # Save best model
    if cider > best_cider:
        best_cider = cider
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss, "cider": best_cider}, best_model_path)
        print(f"New best saved (CIDEr {best_cider:.4f})")

# ================================
# Plotting Results
# ================================
buf = io.BytesIO()
plt.figure(); plt.plot(loss_vals, marker="o"); plt.title("Training Loss"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("training_loss_plot.png"); plt.close(); buf.close()
buf = io.BytesIO()
plt.figure(); plt.plot(cider_vals, label="CIDEr", marker="o"); plt.plot(clip_vals, label="CLIPScore", marker="s")
plt.legend(); plt.title("Validation Scores"); plt.grid(True)
plt.savefig(buf, format="png"); buf.seek(0); Image.open(buf).save("validation_metrics_plot.png")
print("Training complete; metrics logged at", csv_log_path)
