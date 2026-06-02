"""
flash-1-mini — Quickstart: Transformers (bf16, multimodal weights)

Requirements:
    pip install "transformers>=5.5" torch accelerate pillow

Model: simpledirect/flash-1-mini (https://huggingface.co/simpledirect/flash-1-mini)
Architecture: Qwen3_5ForConditionalGeneration — no trust_remote_code needed.

Note on decoding: greedy (do_sample=False) is recommended for legal/regulatory
work to maximise determinism.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "simpledirect/flash-1-mini"

print("Loading processor and model …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")


def generate(messages: list[dict], enable_thinking: bool = False, label: str = "") -> str:
    """Apply chat template, tokenise, generate, and decode."""
    # enable_thinking is passed as a direct kwarg to apply_chat_template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    result = processor.decode(new_ids, skip_special_tokens=True)
    if label:
        print(f"=== {label} ===")
        print(result.strip())
        print()
    return result.strip()


# ---------------------------------------------------------------------------
# 1. English legal prompt — thinking OFF (default for production use)
# ---------------------------------------------------------------------------
en_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Summarise the key obligations imposed on federally regulated employers "
                    "under Part III of the Canada Labour Code regarding unpaid leave for "
                    "family violence victims. Cite the relevant sections."
                ),
            }
        ],
    }
]

generate(en_messages, enable_thinking=False, label="English — thinking OFF (recommended for legal)")

# ---------------------------------------------------------------------------
# 2. French-Canadian legal prompt — thinking OFF
# ---------------------------------------------------------------------------
fr_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Expliquez les obligations principales du vendeur professionnel en vertu "
                    "du contrat de vente au sens du Code civil du Québec, notamment en ce "
                    "qui concerne la garantie légale de qualité. Citez les articles pertinents."
                ),
            }
        ],
    }
]

generate(fr_messages, enable_thinking=False, label="French (Québec civil law) — thinking OFF")

# ---------------------------------------------------------------------------
# 3. Complex reasoning prompt — thinking ON for contrast
#    The model emits <think>…</think> before its visible answer.
#    Useful for hard analytical tasks; strip the think block if you don't
#    want it in the final output.
# ---------------------------------------------------------------------------
reasoning_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "A Canadian corporation incorporated federally under the CBCA wants to "
                    "expand into Québec and conduct regulated financial services. "
                    "Identify the two main regulatory overlaps it must resolve, "
                    "and explain which jurisdiction's rules take precedence in each case."
                ),
            }
        ],
    }
]

generate(reasoning_messages, enable_thinking=True, label="Complex reasoning — thinking ON")
