# flash-1-mini — llama.cpp / GGUF Usage

> **Text-only.** The GGUF variants do not include the vision encoder.
> For image + text inference, use the bf16 weights via Transformers.

## Prerequisites

- A recent llama.cpp build (**with Qwen3.5 hybrid linear-attention support**).
  Builds from mid-2026 onward should include this. If inference produces
  garbage output, your build is too old — rebuild from `main`.
- [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli)
  (`pip install huggingface_hub`)

---

## Available GGUF variants

All files live in the `gguf/` folder of the HF repo:
[huggingface.co/simpledirect/flash-1-mini/tree/main/gguf](https://huggingface.co/simpledirect/flash-1-mini/tree/main/gguf)

| Quantisation | File | Size | Notes |
|---|---|---|---|
| Q6_K | `flash-1-mini-20260602-Q6_K.gguf` | ~3.3 GB | Closest to bf16 quality |
| Q5_K_M | `flash-1-mini-20260602-Q5_K_M.gguf` | ~2.9 GB | Good balance (recommended) |
| Q4_K_M | `flash-1-mini-20260602-Q4_K_M.gguf` | ~2.6 GB | Smallest, slight quality loss |

**Quick guidance:** Use `Q6_K` if VRAM allows and you need the highest fidelity.
Use `Q4_K_M` when fitting into tight RAM/VRAM budgets. `Q5_K_M` is a solid
default for most deployments.

---

## Step 1 — Download a GGUF

```bash
# Install the HF CLI if needed
pip install huggingface_hub

# Download Q5_K_M (recommended)
huggingface-cli download simpledirect/flash-1-mini \
    gguf/flash-1-mini-20260602-Q5_K_M.gguf \
    --local-dir ./flash-1-mini-gguf

# Or Q6_K for higher fidelity
huggingface-cli download simpledirect/flash-1-mini \
    gguf/flash-1-mini-20260602-Q6_K.gguf \
    --local-dir ./flash-1-mini-gguf

# Or Q4_K_M for smallest footprint
huggingface-cli download simpledirect/flash-1-mini \
    gguf/flash-1-mini-20260602-Q4_K_M.gguf \
    --local-dir ./flash-1-mini-gguf
```

---

## Step 2 — Run with llama.cpp

```bash
# Basic completion (thinking disabled via system prompt — or strip <think> blocks)
./llama-cli \
    -m ./flash-1-mini-gguf/flash-1-mini-20260602-Q5_K_M.gguf \
    --ctx-size 8192 \
    --temp 0 \
    --repeat-penalty 1.0 \
    -p "You are a Canadian legal research assistant. Summarise the key employee protections under the Canada Labour Code Part III regarding unjust dismissal."

# Interactive chat mode
./llama-cli \
    -m ./flash-1-mini-gguf/flash-1-mini-20260602-Q5_K_M.gguf \
    --ctx-size 8192 \
    --temp 0 \
    --interactive \
    --chat-template qwen3
```

> **Thinking mode:** The model outputs `<think>…</think>` blocks before the
> visible answer. In llama.cpp you can suppress them by setting
> `--temp 0` and adding a system prompt asking it not to think aloud,
> or by post-processing to strip the blocks from output.

---

## Serving via llama-server (OpenAI-compatible API)

```bash
./llama-server \
    -m ./flash-1-mini-gguf/flash-1-mini-20260602-Q5_K_M.gguf \
    --ctx-size 8192 \
    --host 0.0.0.0 \
    --port 8080

# Then call the API:
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "flash-1-mini",
        "messages": [{"role": "user", "content": "What is the limitation period for contract claims in Ontario?"}],
        "temperature": 0,
        "max_tokens": 512
    }'
```
