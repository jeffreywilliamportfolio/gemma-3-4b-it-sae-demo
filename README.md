# gemma-3-4b-it-sae-demo

Use GemmaScope + Neuronpedia features to steer `gemma-3-4b-it` locally.

This is a practical demo for people who already know local/open-source models,
Hugging Face, quantization, fine-tuning, and inference, but have not used SAEs
before.

The workflow:

```text
Find a feature on Neuronpedia -> run Gemma locally -> /inject or /dim it -> compare outputs
```

## What You Need

```text
Python 3.10+
Enough RAM/unified memory for Gemma 3 4B IT bf16
Hugging Face access to google/gemma-3-4b-it
macOS/Apple Silicon recommended for the current MPS path
```

The demo uses Hugging Face/Transformers bf16 weights, not GGUF/Ollama, because it
needs PyTorch hooks into the model's hidden states.

## 1. Clone

```bash
git clone https://github.com/jeffreywilliamportfolio/gemma-3-4b-it-sae-demo.git
cd gemma-3-4b-it-sae-demo
```

## 2. Install Python Packages

Using a venv is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## 3. Download The Model And SAE Weights

First accept the Gemma terms on Hugging Face:

```text
https://huggingface.co/google/gemma-3-4b-it
```

Then log in if needed:

```bash
huggingface-cli login
```

Download the pinned assets:

```bash
scripts/download_weights.sh
```

This creates:

```text
models/gemma-3-4b-it-hf/
models/gemma-scope-2-4b-it/
```

## 4. Check Setup

```bash
python3 scripts/check_setup.py
```

If this passes, the repo is hydrated correctly.

## 5. Start The Steering Chat

```bash
python3 chat_steer.py
```

Wait for:

```text
ready. /help for commands.
```

## 6. Try A Feature

In the chat:

```text
/clear
/reset
/temp 0.7
/seed 1
/inject 17:4271:1190
Write a short paragraph about silence.
```

Then compare with no steering:

```text
/clear
/reset
/temp 0.7
/seed 1
Write a short paragraph about silence.
```

## 7. Find Your Own Features

Open Neuronpedia:

```text
https://www.neuronpedia.org/gemma-3-4b-it
```

Use these GemmaScope RES-16K sources:

```text
9-gemmascope-2-res-16k
17-gemmascope-2-res-16k
22-gemmascope-2-res-16k
29-gemmascope-2-res-16k
```

Or search from the CLI:

```bash
./scripts/neuronpedia.py search "soft sounds" --res16k --limit 10
./scripts/neuronpedia.py feature 17-gemmascope-2-res-16k 42
```

Neuronpedia IDs map directly to chat commands:

```text
17-gemmascope-2-res-16k / 42 -> /inject 17:42:STRENGTH
```

## Command Cheat Sheet

```text
/inject L:FEATURE:STRENGTH   add a feature direction
/dim L:F1,F2:SCALE           reduce or boost live feature contribution
/clear                       remove active steering
/reset                       clear chat history only
/status                      show active steering/settings
/temp 0.7                    set temperature
/seed 1                      make sampling reproducible
/tokens 150                  set max response tokens
```

Important:

```text
/reset does not clear steering.
/dim and /inject stack until /clear.
Start with small strengths and step up.
```

## Repo Map

```text
chat_steer.py        main interactive demo
scripts/             setup and Neuronpedia helpers
docs/                simple guides
experiments/         optional prior-run scripts
examples/prior-runs/ archived example outputs
models/              downloaded weights live here, ignored by git
probes/              example prompts
```

## Read Next

```text
docs/FEATURE_WORKFLOW.md
docs/CHAT_GUIDE.md
docs/SAE_PRIMER.md
docs/WEIGHTS.md
```

## License

Code in this repo is MIT licensed. Model and SAE weights are downloaded from
Hugging Face and remain under their respective upstream licenses/terms.
