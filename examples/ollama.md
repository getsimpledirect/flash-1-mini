# flash-1-mini — Ollama Usage

> **Text-only.** Ollama loads GGUF files; the GGUF variants do not include
> the vision encoder. For image + text inference use the bf16 weights via
> Transformers (`examples/multimodal_example.py`).

---

## Prerequisites

- [Ollama](https://ollama.com) installed and running.
- `huggingface-cli` for downloading the GGUF
  (`pip install huggingface_hub`).

---

## Step 1 — Download a GGUF

```bash
# Q5_K_M is a good default (2.9 GB)
huggingface-cli download simpledirect/flash-1-mini \
    gguf/flash-1-mini-20260602-Q5_K_M.gguf \
    --local-dir ./flash-1-mini-gguf
```

See [`gguf_llamacpp.md`](./gguf_llamacpp.md) for the full table of quant variants.

---

## Step 2 — Create a Modelfile

Create a file named `Modelfile` in the same directory as the GGUF:

```
FROM ./flash-1-mini-20260602-Q5_K_M.gguf

PARAMETER temperature 0
PARAMETER num_ctx 8192

SYSTEM """You are a bilingual Canadian legal research assistant. You answer in the same language as the user's question."""
```

> **Tip:** Set `temperature 0` for legal work to maximise determinism.

---

## Step 3 — Create the model in Ollama

```bash
# Run from the directory containing both the GGUF and Modelfile
ollama create flash-1-mini -f Modelfile
```

---

## Step 4 — Run

```bash
# Interactive chat
ollama run flash-1-mini

# Single prompt (non-interactive)
ollama run flash-1-mini "What are the notice requirements for terminating a federally regulated employee under the Canada Labour Code?"

# French prompt
ollama run flash-1-mini "Quelles sont les obligations de divulgation d'un vendeur en vertu du Code civil du Québec?"
```

---

## OpenAI-compatible API via Ollama

Ollama exposes an OpenAI-compatible API on port 11434:

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "flash-1-mini",
        "messages": [
            {"role": "user", "content": "Summarise the duty to accommodate under the Canadian Human Rights Act."}
        ],
        "temperature": 0,
        "max_tokens": 512
    }'
```
