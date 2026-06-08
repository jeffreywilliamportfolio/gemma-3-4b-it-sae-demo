# Weight Setup

The repo does not commit model weights, SAE weights, GGUFs, or tensor capture
files. It tracks the exact source repos, pinned revisions, expected paths, and a
download script.

This keeps normal git usable without Git LFS while still making the project
reproducible for someone with enough local GPU/unified-memory headroom.

## Required Sources

Gemma 3 4B instruction-tuned bf16 weights:

```text
repo: google/gemma-3-4b-it
revision: 093f9f388b31de276ce2de164bdc2081324b9767
local path: models/gemma-3-4b-it-hf
official page: https://huggingface.co/google/gemma-3-4b-it
```

GemmaScope 2 4B-IT RES-16K SAE weights:

```text
repo: google/gemma-scope-2-4b-it
revision: 3e94b68be95290aada5b7525cf431d3040f81bb1
local path: models/gemma-scope-2-4b-it
official page: https://huggingface.co/google/gemma-scope-2-4b-it
landing page: https://huggingface.co/google/gemma-scope-2
```

Only these SAE folders are needed for the current scripts:

```text
resid_post/layer_9_width_16k_l0_medium
resid_post/layer_17_width_16k_l0_medium
resid_post/layer_22_width_16k_l0_medium
resid_post/layer_29_width_16k_l0_medium
```

## Download

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

If needed, create `.env` from `.env.example` and set `HF_TOKEN`. Do not commit
`.env`.

Gemma is gated on Hugging Face. Log in and accept the model terms for
`google/gemma-3-4b-it` before downloading.

Then run:

```bash
scripts/download_weights.sh
```

## Expected Layout

After download:

```text
models/gemma-3-4b-it-hf/
  config.json
  generation_config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  tokenizer.json
  tokenizer.model
  tokenizer_config.json

models/gemma-scope-2-4b-it/resid_post/
  layer_9_width_16k_l0_medium/params.safetensors
  layer_17_width_16k_l0_medium/params.safetensors
  layer_22_width_16k_l0_medium/params.safetensors
  layer_29_width_16k_l0_medium/params.safetensors
```

The download script also fetches small config/example files in those SAE
directories when present.

## Why We Do Not Commit Weights

Normal git is the wrong transport for multi-GB model files. This repo is intended
to be cloned, then hydrated locally from the pinned Hugging Face sources above.

No Git LFS is required.
