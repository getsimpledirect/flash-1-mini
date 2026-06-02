# flash-1-mini — Serving Guide

---

## 1. vLLM (bf16, multimodal)

The bf16 weights support both text and image input. vLLM handles multimodal
inference natively for Qwen3.5-based models.

### Install

```bash
pip install vllm
```

### Serve

```bash
vllm serve simpledirect/flash-1-mini \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3
```

**Key flags:**

| Flag | Purpose |
|---|---|
| `--reasoning-parser qwen3` | Separates `<think>…</think>` reasoning from the visible response in the API output |
| `--max-model-len` | Keep this ≤ your GPU VRAM budget; the model supports up to 262 144 tokens |
| `--dtype bfloat16` | Matches training dtype; do not use float16 |

> **Greedy decoding for legal use:** Set `temperature=0` and `top_p=1` in
> your client requests to maximise determinism — important for citation
> reliability.

### Example client call (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="simpledirect/flash-1-mini",
    messages=[
        {
            "role": "user",
            "content": "What are the requirements for a valid statutory declaration under the Canada Evidence Act?"
        }
    ],
    temperature=0,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### Multimodal (image + text) via vLLM

Pass an image URL or base64-encoded image in the `content` list:

```python
response = client.chat.completions.create(
    model="simpledirect/flash-1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/document.png"}},
                {"type": "text", "text": "Identify the document type and extract the key obligations."},
            ],
        }
    ],
    temperature=0,
    max_tokens=512,
)
```

> Multimodal inference requires the bf16 weights. GGUF variants are text-only.

---

## 2. Ollama (GGUF, text-only)

For deployments where running the full bf16 model is impractical, Ollama
provides a straightforward path using the quantised GGUF files.

See [`../examples/ollama.md`](../examples/ollama.md) for the complete setup
walkthrough: downloading a GGUF, writing a Modelfile, and running
`ollama create` / `ollama run`.

**Note:** Ollama/GGUF does **not** support image input. Use vLLM + bf16 for
multimodal workloads.

---

## Hardware guidance

| Setup | Recommended variant | Approx. VRAM |
|---|---|---|
| GPU ≥ 24 GB | bf16 via vLLM | ~10 GB at 32 k ctx |
| GPU 8–16 GB | Q6_K GGUF via Ollama/llama.cpp | ~4 GB |
| GPU 4–8 GB / CPU | Q4_K_M GGUF via Ollama/llama.cpp | ~3 GB |
