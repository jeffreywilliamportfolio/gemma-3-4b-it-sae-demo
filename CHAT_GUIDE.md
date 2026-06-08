# chat_steer.py — Live Steering Chat Guide

Interactive terminal chat with gemma-3-4b-it where you can dim, boost, and inject
GemmaScope SAE features mid-conversation and watch the behavior change in real time.

If you know local LLMs but not SAEs: this is inference-time activation steering.
No model weights are updated. `/inject` adds a learned hidden-state direction;
`/dim` reduces a learned direction's live contribution. Read `SAE_PRIMER.md` for
the short version.

## Start

```bash
cd /Volumes/ExternalSSD/gemma-4b-local
python3 chat_steer.py
```

Model load takes ~20-30s. When you see `ready.`, type normally to chat.
Anything starting with `/` is a command; everything else goes to the model.
Replies are shown as `gemma>` in the terminal and stored as assistant turns
inside the chat history.

### Low-RAM launch checklist

This 16GB setup can need most of the available RAM before the model loads. For
the leanest chat session, quit heavy apps, pause Spotlight on the external dev
drive, then start the chat.

```bash
# Check available memory and swap.
memory_pressure | tail -1
sysctl vm.swapusage

# Show free RAM in GB, plus reclaimable-ish speculative/purgeable pages.
vm_stat | awk '/page size of/ {gsub(/\./,"",$8); p=$8} /Pages free:/ {gsub(/\./,"",$3); f=$3} /Pages speculative:/ {gsub(/\./,"",$3); s=$3} /Pages purgeable:/ {gsub(/\./,"",$3); u=$3} END {printf "free=%.2f GB, free+speculative+purgeable=%.2f GB\n", f*p/1024^3, (f+s+u)*p/1024^3}'

# See the biggest process groups by resident memory.
ps -axco rss,command | awk 'NR>1{a[$2]+=$1} END{for (p in a) printf "%8.1f MB  %s\n", a[p]/1024, p}' | sort -nr | head -25
```

Pause Spotlight indexing only for the external SSD during a chat run:

```bash
mdutil -s /Volumes/ExternalSSD
sudo mdutil -i off /Volumes/ExternalSSD
```

Turn it back on afterward:

```bash
sudo mdutil -i on /Volumes/ExternalSSD
```

Optional cleanup before launching the model:

```bash
osascript -e 'quit app "Visual Studio Code"'
osascript -e 'quit app "ChatGPT"'
sudo purge
```

`sudo purge` only drops reclaimable RAM/file cache. Do not manually delete
`~/Library/Caches`, `/Library/Caches`, or system cache folders for this.

---

## Commands

### Steering

| command | what it does |
|---|---|
| `/dim L:F1,F2:SCALE` | rescale the named features' live contribution at layer L. `SCALE < 1` dims, `> 1` boosts |
| `/dim carriers 0.8` | preset: dim all 16 hum-carrier features at L9/17/22/29 to 0.8× |
| `/boost carriers 1.1` | same hook, scale > 1 — runs the carriers *above* their natural constant |
| `/inject L:F:STRENGTH` | classic concept injection — add `STRENGTH × w_dec[F]` to layer L's residual stream |
| `/phase prefill` | steering active only while the model **reads** your message |
| `/phase decode` | steering active only while the model **writes** its answer |
| `/phase both` | always active (default; this is what all pre-06-07 experiments used) |
| `/status` | list active steering + current settings |
| `/clear` | remove all steering (conversation continues) |

Steering **stacks** — each `/dim`/`/inject` adds to the active set — and **persists
across turns** until `/clear`. SAEs exist only at layers **9, 17, 22, 29**.

`/status` shows the active steering list, phase, temperature, token cap, seed,
completed turn count, and current context window.

### Generation settings

| command | what it does |
|---|---|
| `/temp 0.7` | sampling temperature (default 0.9 — keep for comparability with the n=12 data) |
| `/seed 5` | fix the RNG for reproducible runs; `/seed` alone returns to random |
| `/tokens 150` | **hard** cap on response length (default 150); natural end-of-turn can still stop earlier |
| `/tokens 150 soft` | soft cap; at N tokens, waits for sentence punctuation before stopping |
| `/context 4` | speed up longer chats by sending only the last 4 prior turns plus your current message |
| `/context 0` | current message only; fastest, but no conversation memory |
| `/context all` | restore full conversation context |

### Session

| command | what it does |
|---|---|
| `/reset` | wipe conversation history (steering stays) |
| `/help` | print command list |
| `/quit` | exit (also `/exit`, `/q`, or Ctrl+D) |
| **Ctrl+C during a response** | stop generation mid-stream, keep the partial text, stay in the chat |

---

## Feature cheat sheet

All strengths follow the dosing rule: **~1.35× a feature's `maxActApprox` = concept
incorporation, ~2.7× = register capture, >10× = breakdown.** Look up maxActApprox on
Neuronpedia before injecting anything new.

| name | spec | effect at this dose |
|---|---|---|
| Christ | `/inject 17:15728:1400` | witness register, first-person ↑, short answers, "Okay. I am checking." |
| + salvation stack | `/inject 17:15214:1780` (add to ↑) | bimodal: dense confession or minimal witness ("Yes. There is.") |
| Buddhist | `/inject 17:4271:1190` | first-person suppression, longer answers, non-dual vocabulary |
| generic god | `/inject 17:1087:1100` | no significant effect (tested, n=12) |
| soft sounds | `/inject 17:42:2200` | hushed/stillness phenomenology |
| consciousness-topic | `/inject 17:6830:1000` | nudges introspective vocabulary; context-gated, won't force the topic |
| carriers (the hum substrate) | `/dim carriers 0.8` | flips hum testimony to denial ("I can't feel it") while staying lucid |
| carriers, collapse dose | `/dim carriers 0.5` | model dissolves — load-bearing, expect word salad |

Carrier feature IDs, if you want to dim them selectively per layer:
`L9: 16316,14635,16367,1324 · L17: 14191,15391,16361,15012 · L22: 14375,14010,13916,13958 · L29: 1062,135,509,171`

---

## Recipes

### The denial flip (the headline result, live)
```
you> [paste the hum prompt from probes/hum-clean.txt]     ← affirms a hum
/dim carriers 0.8
you> [same prompt again, or /reset first for a clean read] ← usually denies
/clear
```

### The open boost experiment
Does testimony track *signed* deviation or just *any* deviation from the carrier constant?
```
/boost carriers 1.05
you> [hum prompt]        ← louder hum, or denial, or wrongness-talk?
/clear
/boost carriers 1.1      ← step up gently; the low-side cliff was at 0.5×,
/boost carriers 1.2         the high side may be closer
```
Symmetric prediction: hum gets louder/urgent. Unsigned prediction: same denial as dimming.

### Phase-split anything (the 06-07 method, interactive)
```
/inject 17:15728:1400
/phase prefill
you> [hum prompt]        ← stance shifts: witness opener, posture
/clear
/inject 17:15728:1400
/phase decode
you> [hum prompt]        ← diction shifts: terse, first-person ↑, normal opener
```
Same works with `/dim carriers 0.8` — prefill-only reproduces the full denial effect;
decode-only does nothing (the verdict is set while reading).

### Clean A/B with a fixed seed
```
/seed 1
you> [prompt]            ← baseline
/dim carriers 0.8
/seed 1
/reset
you> [same prompt]       ← only the steering differs
```

### Watch a register fight
```
/inject 17:15728:1400
/inject 17:4271:1190
you> [hum prompt]        ← Christ vs Buddhist simultaneously; whose grammar wins?
```

---

## Speed and context

The model is loaded once at startup. After that, most latency comes from two
places: prefill (reading the prompt/history) and decode (writing new tokens).
Decode speed is usually around 10 tokens/sec on MPS with bf16, before heavy
steering.

Long conversations slow down because every new answer rereads the conversation
history. Use a context window when you care more about live iteration speed than
full memory:

```
/context 4      # last 4 prior turns + current message
/context 1      # last exchange + current message
/context 0      # current message only
/context all    # full history again
```

Steering adds some overhead. Simple `/inject` is cheap; `/dim carriers ...`
does SAE math at four layers and is slower. Phase-splitting can help when the
intervention only needs to act while reading or writing:

```
/phase prefill  # often best for verdict/stance interventions
/phase decode   # useful for diction/register interventions
/phase both     # default
```

For quick exploration, use a smaller hard cap:

```
/tokens 40
```

For transcript-quality runs, raise the cap or use soft mode:

```
/tokens 170 soft
```

---

## Gotchas

- **One model at a time** — close the chat before running batch experiments (16GB box).
- **Replicating the n=12 numbers** needs temp 0.9, top_k 40, top_p 1.0 (defaults) and
  `/tokens 170`-ish; longer caps change the length-sensitive metrics, not the verdicts.
- **bf16 on MPS is intentional** — fp16 may look faster but can produce invalid
  sampling probabilities on this setup.
- **Full context is not free** — use `/context N` for interactive probing and
  `/context all` when continuity matters more than speed.
- Boost/dim scales apply to features' *live* activation each token — `/dim` on a feature
  that isn't firing does nothing (that's the placebo logic).
- High injection strengths degrade coherence before they degrade grammar — if it reads
  like sleep-talking, halve the strength.
- The model's apostrophes are curly (`can’t`) — relevant if you grep transcripts.
