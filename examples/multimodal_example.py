"""
flash-1-mini — Multimodal Example: image + text input

Requirements:
    pip install "transformers>=5.5" torch accelerate pillow

IMPORTANT: This example requires the bf16 weights (simpledirect/flash-1-mini on
Hugging Face). The GGUF variants in the gguf/ folder are TEXT-ONLY and cannot
accept image input — they cannot be used for multimodal inference.

Usage:
    python multimodal_example.py                  # generates a test image
    python multimodal_example.py path/to/image.jpg  # uses your own image
"""

import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "simpledirect/flash-1-mini"

# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def make_test_image() -> Image.Image:
    """Create a simple synthetic document image for testing."""
    img = Image.new("RGB", (640, 160), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 620, 140], outline=(0, 0, 0), width=2)
    draw.text(
        (40, 40),
        "LEASE AGREEMENT (EXCERPT)\n\n"
        "Tenant shall provide 60 days written notice of termination.\n"
        "Monthly rent: $2,400 CAD. Jurisdiction: Ontario.",
        fill=(0, 0, 0),
    )
    return img


if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
else:
    print("No image path provided — generating a synthetic test document image.")
    image = make_test_image()

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------

print("Loading processor and model …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Multimodal inference
# ---------------------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "This is an excerpt from a legal document. "
                    "Identify the document type, extract the key obligations mentioned, "
                    "and note the applicable jurisdiction."
                ),
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = processor(
    text=text,
    images=[image],          # pass image list here
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
response = processor.decode(new_ids, skip_special_tokens=True)

print("=== Model response ===")
print(response.strip())
