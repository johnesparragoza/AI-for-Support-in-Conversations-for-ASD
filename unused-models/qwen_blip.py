import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor,
    Blip2ForConditionalGeneration, Blip2Processor,
    StoppingCriteria, StoppingCriteriaList
)
import textwrap

#Attempt to combine BLIP-2 and Qwen/Qwen2-VL-7B-Instruct to generate different caption levels
#Part of the project is adapt to the individual's communication level.
#BLIP-2 served as the primary caption (Level 4-Caption)
#Qwen-Instruct model would try to generate different captions based on the Level-4 caption and the image uploaded
#Prompt Levels 1-2 and 4 have no issues; issues with caption levels 3 and 5-6. Need to be fine tuned.

# -------- Settings --------
folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
font = ImageFont.load_default()

# -------- Caption Prompts --------
caption_level_prompts = [
    "Identify the primary subject with a single noun. (e.g., 'Dog.', 'Park.', 'Ball.').",  # Level 1
    "Write a simple 2 word preset tense sentence stating what the subject is doing, subject + verb (e.g., 'Brids sings.', 'People dance.').",  # Level 2
    "Write a four‑word caption: subject + verb + adverb(e.g., If the previous caption was 'The dog runs', the new caption could be 'A brown dog runs.').",  # Level 3
    "BLIP caption placeholder",  # Level 4 (BLIP)
    "Building on the BLIP caption (Level 4), add a new clause  that describes a specific, observable detail about the subject's key distinguishing feature. This new clause should describe a specific, visible detail in the image, enhancing the overall description(e.g., If BLIP was 'A dog leaps across the grass', the new caption could be 'A dog is leaping across the grass, its paws extended.' OR if BLIP was 'Two knights fight in the desert', the new caption could be 'Two knights engage in a duel in the desert, one with a blue plume and the other with a red one.').",  # Level 5
    "Incoporating and building on the Level 5 caption, complete the description by adding a second, separate independent clause with 'and'. This new clause should describe a detail about the surrounding environment, weather, or background.", # Level 6
    "Craft a cohesive narrative sentence that seamlessly integrates all descriptive elements from Level 6, ensuring the sentence is grammatically correct, flows naturally, and encapsulates the essence of the image in a concise and engaging manner." # Level 7
]

# -------- Helper Functions --------
def clean_intro(text):
    starters = [
        r"^(this image|the image|this photo|the photo|it depicts|depicts|shows|image of|picture of|photo of)\b[:,\s]*"
    ]
    text = text.strip()
    for pattern in starters:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    incomplete_endings = [r"\s+and$", r"\s+with$", r"\s+but$", r"\s+so$", r"\s+because$"]
    for ending in incomplete_endings:
        text = re.sub(ending, "", text, flags=re.IGNORECASE).strip()

    text = re.sub(r"\s+", " ", text)
    return text

def smart_trim(text, max_words=25):
    words = text.split()
    if len(words) <= max_words:
        return text

    trimmed = " ".join(words[:max_words])
    last_punct = max(trimmed.rfind(p) for p in [".", ",", ";"])
    if last_punct != -1 and last_punct > max_words // 2:
        trimmed = trimmed[:last_punct+1]

    if not trimmed.endswith((".", ",", ";")):
        trimmed += "..."

    return trimmed.strip()

def capitalize_and_punctuate(text):
    text = text.strip()
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith(('.', '!', '?')):
        text += "."
    return text

# -------- Stopping Criteria --------
class SentenceEndStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, end_punctuations={'.', '!', '?'}):
        self.tokenizer = tokenizer
        self.end_punct_ids = set()
        for p in end_punctuations:
            token_id = tokenizer.convert_tokens_to_ids(p)
            if token_id is not None:
                self.end_punct_ids.add(token_id)

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if the last generated token is a sentence-ending punctuation token
        return input_ids[0, -1].item() in self.end_punct_ids

# -------- Load Models --------
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch_dtype, device_map="auto"
)
qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map="auto"
)
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Prepare stopping criteria instance once
stopping_criteria = StoppingCriteriaList([SentenceEndStoppingCriteria(qwen_proc.tokenizer)])

# -------- Image Processing --------
for filename in os.listdir(folder):
    if not filename.lower().endswith(image_exts):
        continue

    qwen_captions = []  # Reset captions per image
    level4_caption = ""  # Reset BLIP caption per image

    image_path = os.path.join(folder, filename)
    image = Image.open(image_path).convert("RGB")

    # -- Resize for consistency --
    max_width = 512
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height))

    # -- Get BLIP-2 caption (Level 4) --
    blip_inputs = {k: v.to(device) for k, v in blip_processor(images=image, return_tensors="pt").items()}
    blip_out = blip_model.generate(**blip_inputs)
    blip_caption = blip_processor.decode(blip_out[0], skip_special_tokens=True).strip()
    blip_caption = capitalize_and_punctuate(blip_caption)
    level4_caption = f"BLIP-2 (Level 4): {blip_caption}"

    # -- Get Qwen captions (Levels 1–3, 5–7) --
    for idx, prompt in enumerate(caption_level_prompts):
        if idx == 3:
            continue  # Skip Level 4 (BLIP)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt}\nBLIP caption: {blip_caption}"}
                ],
            }
        ]
        text_prompt = qwen_proc.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = qwen_proc(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(device)

        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=640,  # Adjust max tokens as needed
            eos_token_id=qwen_proc.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            pad_token_id=qwen_proc.tokenizer.pad_token_id,
            do_sample=False,  # Set True for sampling if desired
        )
        # Extract only the generated portion, excluding prompt tokens
        generated_ids = [
            output_ids[i, len(inputs.input_ids[i]):] for i in range(len(output_ids))
        ]
        decoded = qwen_proc.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # If caption ends with comma or ellipsis, attempt to complete sentence by generating more tokens
        if decoded.endswith(",") or decoded.endswith(", ") or decoded.endswith("..."):
            more_output_ids = qwen_model.generate(
                output_ids,  # Use previous outputs as the prefix input_ids
                max_new_tokens=620,
                eos_token_id=qwen_proc.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                pad_token_id=qwen_proc.tokenizer.pad_token_id,
                do_sample=False,
            )
            # Slice off previously generated tokens to get only new tokens, decode and append
            new_tokens = more_output_ids[:, output_ids.shape[1]:]
            additional_text = qwen_proc.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()
            decoded += " " + additional_text
            decoded = decoded.strip()

        # Post-processing per level
        if idx == 0:
            # Level 1: Extract the noun only (last word before a comma)
            decoded = decoded.split(",")[0].split(" ")[-1]
        elif idx == 1:
            # Level 2: Ensure exactly two words: subject + verb present tense
            words = decoded.split()
            if len(words) >= 2:
                decoded = f"{words[0]} {words[1]}"
        else:
            # For other levels, clean intro, trim length, capitalize and punctuate
            decoded = clean_intro(decoded)
            decoded = smart_trim(decoded, max_words=25)
            decoded = capitalize_and_punctuate(decoded)

        qwen_captions.append(f"Qwen2VL (Level {idx + 1}): {decoded}")

    # -- Combine captions --
    captions = qwen_captions[:3] + [level4_caption] + qwen_captions[3:]

    # -- Compose new image with captions --
    new_img = Image.new("RGB", (image.width, image.height + 200), "white")
    new_img.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    y_offset = image.height + 5
    for cap in captions:
        for line in textwrap.wrap(cap, width=80):
            draw.text((5, y_offset), line, fill="black", font=font)
            y_offset += 12

    out_path = os.path.join(output_folder, filename)
    new_img.save(out_path)
    print(f"✅ Saved captioned image: {out_path}")
