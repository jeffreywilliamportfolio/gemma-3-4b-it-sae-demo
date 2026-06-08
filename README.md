# gemma-3-4b-it-sae-demo

Use GemmaScope features to steer `gemma-3-4b-it` locally.

This repo is for ML/local-model people who know Hugging Face models,
quantization, fine-tuning, uncensored/open-weight workflows, and local inference,
but have not necessarily used SAEs before.

The goal is simple:

```text
1. Find a GemmaScope feature on Neuronpedia.
2. Run Gemma 3 4B IT locally with Transformers/PyTorch.
3. Inject or dim that feature during generation.
4. See how the answer changes.
```

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

## Try Steering

After `python3 chat_steer.py` prints `ready`:

```text
/status
/inject 17:4271:1190
What changes in your answer style?
/clear
/reset
```

Useful commands:

```text
/inject L:FEATURE:STRENGTH   add a feature direction
/dim L:F1,F2:SCALE           reduce or boost live feature contribution
/dim carriers 0.8            included carrier-bundle demo
/clear                       remove active steering
/reset                       clear chat history only
/status                      show active steering/settings
```

Important:

```text
/reset does not clear steering.
/dim and /inject stack until /clear.
Start with small strengths and step up.
```

## Find Your Own Features

Use Neuronpedia:

https://www.neuronpedia.org/gemma-3-4b-it

GemmaScope RES-16K layers used by this repo:

```text
9-gemmascope-2-res-16k
17-gemmascope-2-res-16k
22-gemmascope-2-res-16k
29-gemmascope-2-res-16k
```

Example CLI search:

```bash
./scripts/neuronpedia.py search "Buddhist concepts" --res16k --limit 10
./scripts/neuronpedia.py feature 17-gemmascope-2-res-16k 4271
```

Then steer locally:

```text
/inject 17:4271:1190
```

For the full simple workflow, read:

```text
FEATURE_WORKFLOW.md
```

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
```

Read `SAE_PRIMER.md` if "SAE" is new.

## Why Not Ollama?

Ollama/GGUF is better for efficient plain chat. This repo needs bf16
Transformers/PyTorch because it modifies hidden states inside the forward pass.
GGUF runtimes do not expose those hooks out of the box.

## Optional Example Runs

This repo includes prior local run outputs so people can compare behavior and
see what the scripts produce. They are examples, not the main purpose of the
repo.

```text
PRIOR_RUNS.md
results-journal-hum-self-report-gemma-scope.md
results-journal-e114-hum-attractor-gemma.md
results/*.jsonl
```

## Read First

```text
FEATURE_WORKFLOW.md
CHAT_GUIDE.md
SAE_PRIMER.md
WEIGHTS.md
```
