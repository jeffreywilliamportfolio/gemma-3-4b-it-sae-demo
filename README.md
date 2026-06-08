# Gemma 4B Local SAE Steering

Local `gemma-3-4b-it` experiments with GemmaScope 2 residual-stream SAEs. The
main interface is `chat_steer.py`, an interactive terminal chat that can dim,
boost, and inject SAE features while the model is generating.

The release-grade result is carrier dimming on the hum prompt: dimming a specific
tonic carrier bundle to `0.8x` moves many runs away from confident hum testimony
toward denial or epistemic humility, while matched controls do not. See
`RESULTS.md` for the reproducible summary.

## Setup

```bash
python3 -m pip install -r requirements.txt
scripts/download_weights.sh
```

The model and SAE weights are not committed. `scripts/download_weights.sh` pulls
the pinned Hugging Face revisions documented in `WEIGHTS.md`.

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
RESULTS.md
CHAT_GUIDE.md
WEIGHTS.md
EXPLANATION.md
results-journal-hum-self-report-gemma-scope.md
```
