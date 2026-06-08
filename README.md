# Gemma 4B Local SAE Steering

Local `gemma-3-4b-it` experiments with GemmaScope 2 SAE feature steering. This
repo is written for people who already know local LLMs, Hugging Face models,
quantization, fine-tuning, and open-weight workflows, but may not have used SAEs
before.

The main interface is `chat_steer.py`, an interactive terminal chat that can
dim, boost, and inject learned hidden-state feature directions while the model is
generating.

The release-grade result is carrier dimming on the hum prompt: dimming a specific
tonic carrier bundle to `0.8x` moves many runs away from confident hum testimony
toward denial or epistemic humility, while matched controls do not. See
`RESULTS.md` for the reproducible summary.

## What This Is

This is not a fine-tune, LoRA, jailbreak, or prompt pack. It leaves the model
weights frozen and modifies selected internal activations during inference.

Plain local-model translation:

```text
SAE feature = a learned hidden-state direction from GemmaScope
inject      = add that direction during a forward pass
dim         = reduce that direction's live contribution during a forward pass
carrier     = a feature bundle whose dimming changed the target behavior
```

Read `SAE_PRIMER.md` first if "SAE" is new but you already understand local model
runtimes.

## Setup

```bash
python3 -m pip install -r requirements.txt
scripts/download_weights.sh
```

The model and SAE weights are not committed. `scripts/download_weights.sh` pulls
the pinned Hugging Face revisions documented in `WEIGHTS.md`.

This repo uses bf16 Transformers/PyTorch so it can hook hidden states. Ollama or
GGUF runtimes are useful for plain chat, but they cannot run these interventions
without a custom backend.

## Run Interactive Chat

```bash
python3 chat_steer.py
```

Useful baseline ritual:

```text
/clear
/reset
/status
```

Carrier dimming:

```text
/dim carriers 0.8
```

Important: `/reset` clears history only. Use `/clear` to remove steering.

## Reproduce Main n=12 Carrier-Dimming Run

```bash
python3 experiment_dim_n12.py
```

Outputs append to:

```text
results/dim_n12.jsonl
```

## Read First

```text
SAE_PRIMER.md
RESULTS.md
CHAT_GUIDE.md
WEIGHTS.md
EXPLANATION.md
results-journal-hum-self-report-gemma-scope.md
```
