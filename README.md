# flash-1-mini

**A compact, bilingual, vision-capable model specialized for Canadian legal and regulatory work — in English and Canadian French.**

[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-simpledirect%2Fflash--1--mini-yellow)](https://huggingface.co/simpledirect/flash-1-mini)
[![License: MIT](https://img.shields.io/badge/repo%20code-MIT-blue)](LICENSE)
[![Base model: Qwen3.5-4B](https://img.shields.io/badge/base-Qwen3.5--4B%20(Apache--2.0)-orange)](https://huggingface.co/Qwen/Qwen3.5-4B)

---

## What it is

flash-1-mini is a 4.54B-parameter fine-tune of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
trained by Alpine Pacific Trading Inc. (operating as SimpleDirect®) for Canadian legal and regulatory
workflows. It is strong at correctly-formatted legal citations and instruction-following, bilingual
in English and Canadian French, and covers both common law and Québec civil law traditions. It
retains the base model's general reasoning and vision capabilities.

**This repository contains code and documentation only.** Model weights live on Hugging Face:
[huggingface.co/simpledirect/flash-1-mini](https://huggingface.co/simpledirect/flash-1-mini)

---

## Highlights

Numbers are flash-1-mini vs. Qwen3.5-4B (base), measured under identical conditions.
Full methodology and the CBLRE eval-set documentation are in the [Hugging Face model card](https://huggingface.co/simpledirect/flash-1-mini).

| Metric | flash-1-mini | Base (Qwen3.5-4B) | Delta |
|---|---|---|---|
| **Legal citation integrity** (CBLRE citation) | **42.1%** | 15.8% | +26.3 pp |
| **Instruction-following** (IFEval prompt-strict) | **53.2%** | 30.3% | +22.9 pp |
| **Bilingual parity** (privacy-compliance EN/FR) | **90.9% / 90.9%** | — | ratio 1.00 |
| English legal — international law (MMLU) | **76.0%** | 70.3% | +5.7 pp |
| General knowledge (MMLU) | ~69.8% | ~69.8% | ≈ 0 |
| Complex reasoning (BBH) | **79.0%** | 68.6% | +10.4 pp |
| Vision-capable (image + text) | Yes | Yes | — |

Key takeaways:
- **2.7× more reliable legal citations** (42.1% vs 15.8%)
- **+22.9 points instruction-following** — more likely to comply with format and scope constraints
- **Bilingual parity ratio 1.00** — identical privacy-compliance accuracy in English and French
- General knowledge unchanged; complex reasoning substantially improved

> **Note on licenses:** This repository (code, examples, documentation) is MIT.
> The flash-1-mini model weights, hosted on Hugging Face, are Apache 2.0 (as a
> derivative of Qwen3.5-4B). When citing the model, use Apache 2.0.

---

## Where it's weaker

Honest disclosure of verified regressions vs. the base model (identical conditions):

| Metric | flash-1-mini | Base (Qwen3.5-4B) | Delta |
|---|---|---|---|
| Retrieval / RAG | 75.5% | 80.5% | −5.0 pp |
| Function-calling (BFCL v4) | 28.6% | 37.7% | −9.1 pp |
| French professional-law MCQ (Global-MMLU FR) | 44.6% | 49.0% | −4.4 pp |
| CBLRE Québec civil law | 90.0% | 95.0% | −5.0 pp |

If your workflow is primarily retrieval-augmented generation, tool-calling, French professional-law
MCQ, or Québec civil law, benchmark carefully before deploying over the base model.

---

## Quick start

### Transformers (bf16 — text + image)

```bash
pip install "transformers>=5.5" torch accelerate pillow
```

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("simpledirect/flash-1-mini")
model = AutoModelForImageTextToText.from_pretrained(
    "simpledirect/flash-1-mini",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # no trust_remote_code needed
)

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Summarise the key protections under the Canada Labour Code Part III."}],
    }
]

# enable_thinking=False passed directly to apply_chat_template (not via chat_template_kwargs)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0)

new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
print(processor.decode(new_ids, skip_special_tokens=True))
```

See [`examples/quickstart_transformers.py`](examples/quickstart_transformers.py) for English + French
prompts and a thinking-on example.

### GGUF / Ollama (text-only)

```bash
# Download
huggingface-cli download simpledirect/flash-1-mini \
    gguf/flash-1-mini-20260602-Q5_K_M.gguf --local-dir ./flash-1-mini-gguf

# Ollama
ollama create flash-1-mini -f Modelfile
ollama run flash-1-mini "What are the notice requirements under the Canada Labour Code?"
```

See [`examples/gguf_llamacpp.md`](examples/gguf_llamacpp.md) and
[`examples/ollama.md`](examples/ollama.md) for full instructions.

---

## Models

| Variant | Modalities | Size | Location |
|---|---|---|---|
| bf16 (full weights) | **Text + image** | ~9 GB | [HF main repo](https://huggingface.co/simpledirect/flash-1-mini) |
| Q6_K GGUF | Text only | ~3.3 GB | [gguf/ folder](https://huggingface.co/simpledirect/flash-1-mini/tree/main/gguf) |
| Q5_K_M GGUF | Text only | ~2.9 GB | [gguf/ folder](https://huggingface.co/simpledirect/flash-1-mini/tree/main/gguf) |
| Q4_K_M GGUF | Text only | ~2.6 GB | [gguf/ folder](https://huggingface.co/simpledirect/flash-1-mini/tree/main/gguf) |

**GGUFs are text-only** — they do not include the vision encoder. For image + text
inference, use the bf16 weights.

---

## Thinking mode

The model **thinks by default**, emitting `<think>…</think>` before its visible answer.
For production legal use, disable thinking for deterministic output:

```python
# Direct kwarg — do NOT wrap in chat_template_kwargs
text = processor.apply_chat_template(messages, ..., enable_thinking=False)
```

Thinking ON is useful for complex multi-step reasoning (e.g., cross-jurisdictional analysis).

---

## Technical details

| Property | Value |
|---|---|
| Architecture | Qwen3_5ForConditionalGeneration |
| Parameters | 4.54B |
| dtype | bfloat16 |
| Hidden size | 2560 |
| Layers | 32 |
| Attention heads | 16 |
| Vocabulary | 248 320 |
| Context length | 262 144 |
| Tied embeddings | Yes |
| Transformers requirement | >= 5.5 (native; no `trust_remote_code`) |

---

## Intended use & responsible use

flash-1-mini is designed as a **drafting and research assistant** for Canadian legal and
regulatory workflows — summarisation, citation lookup, bilingual Q&A, document analysis,
and compliance checking.

**It assists; it does not replace professional judgment.**

- Always **verify citations against primary sources** before relying on them.
  Citation accuracy improved substantially over base, but is not perfect.
- This model does **not** provide legal advice and should not be presented to clients
  as doing so.
- Outputs may contain errors, outdated information, or jurisdiction mismatches.
  Human review is required before any use in professional or legal contexts.

---

## License

**Repository code** (this repo): [MIT](LICENSE) — Copyright © 2026 Alpine Pacific Trading Inc. (operating as SimpleDirect®)

**Model weights**: Apache-2.0 — derivative of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
See the [model's LICENSE](https://huggingface.co/simpledirect/flash-1-mini/blob/main/LICENSE) and
[NOTICE](https://huggingface.co/simpledirect/flash-1-mini/blob/main/NOTICE) on Hugging Face.

Full benchmark methodology, evaluation harness details, and CBLRE eval-set documentation
are available in the [Hugging Face model card](https://huggingface.co/simpledirect/flash-1-mini).
Training code, data, and hyperparameter recipes are intentionally not published.
