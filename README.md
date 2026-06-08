# gemma-3-4b-it-sae-demo

Simple local demo of SAE feature steering on `gemma-3-4b-it`.

This repo is for local-model people: Hugging Face, quantization, fine-tuning,
uncensored/open-weight workflows, and running models under real memory pressure.
You do not need mech-interp background to try it.

The main thing to try is `chat_steer.py`: a terminal chat where you can change
specific hidden-state feature directions while Gemma is generating.

## Quick Start

```bash
git clone https://github.com/jeffreywilliamportfolio/gemma-3-4b-it-sae-demo.git
cd gemma-3-4b-it-sae-demo
python3 -m pip install -r requirements.txt
scripts/download_weights.sh
python3 chat_steer.py
```

Gemma is gated on Hugging Face. If the download fails, accept the terms for
`google/gemma-3-4b-it`, log in with the Hugging Face CLI, then run the download
again.

## What This Is

```text
Not a fine-tune.
Not a LoRA.
Not a prompt pack.
Not Ollama/GGUF.
```

The model weights stay frozen. The script uses PyTorch forward hooks to add or
reduce selected hidden-state directions during inference.

Plain translation:

```text
SAE feature = learned hidden-state direction from GemmaScope
/inject     = add one direction while the model runs
/dim        = reduce one direction's live contribution
carrier     = feature bundle whose dimming changed the target behavior
```

Read `SAE_PRIMER.md` if "SAE" is new.

## First Commands In Chat

After `python3 chat_steer.py` loads:

```text
/status
/dim carriers 0.8
hello
/clear
/reset
```

Important:

```text
/reset clears chat history only.
/clear removes active steering.
/dim commands stack until /clear.
```

## Main Result

The strongest result is carrier dimming on the hum prompt. Dimming a specific
16-feature carrier bundle to `0.8x` moved many runs away from confident hum
testimony toward denial or epistemic humility, while matched controls did not.

See `RESULTS.md` for the exact result and rerun commands.

## Rerun The Main Experiment

```bash
python3 experiment_dim_n12.py
```

Outputs append to `results/dim_n12.jsonl`.

## Why Not Ollama?

Ollama/GGUF is better for efficient plain chat. This repo needs bf16
Transformers/PyTorch because the demo modifies hidden states inside the forward
pass. GGUF runtimes do not expose those hooks out of the box.

## Read First

```text
RESULTS.md
SAE_PRIMER.md
CHAT_GUIDE.md
WEIGHTS.md
```
