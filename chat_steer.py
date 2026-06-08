#!/usr/bin/env python3
"""Interactive terminal chat with gemma-3-4b-it + live SAE steering REPL.

Usage:  python3 chat_steer.py

Slash commands (anything else is sent to the model as chat):
  /dim L:F1,F2,..:SCALE   rescale named features' live contribution at layer L
                          (SCALE < 1 dims, > 1 BOOSTS — same hook, same math)
  /dim carriers 0.8       preset: the 16 hum-carrier features at L9/17/22/29
  /boost carriers 1.2     alias for /dim with a >1 scale
  /inject L:F:STRENGTH    add STRENGTH x w_dec[F] at layer L (concept injection)
  /phase prefill|decode|both   when steering is active (default both)
  /status                 show active steering
  /clear                  remove all steering
  /reset                  wipe conversation history
  /context N|all          send only last N prior turns plus current user message
  /seed N | /temp X
  /tokens N              hard cap on response tokens (default 150)
  /tokens N soft         soft cap: at N tokens, finish the sentence then stop
  /quit
"""
import re
import sys
import threading
from queue import Empty
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

ROOT = Path(__file__).parent
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
SAE_LAYERS = {9, 17, 22, 29}

CARRIERS = {9: [16316, 14635, 16367, 1324], 17: [14191, 15391, 16361, 15012],
            22: [14375, 14010, 13916, 13958], 29: [1062, 135, 509, 171]}

C_USER, C_BOT, C_SYS, C_ERR, C_END = "\033[36m", "\033[35m", "\033[33m", "\033[31m", "\033[0m"

device, dtype = "mps", torch.bfloat16
print(f"{C_SYS}loading gemma-3-4b-it (bf16, mps)...{C_END}", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()
layers = model.model.language_model.layers


def as_int_list(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(x) for x in value]


END_OF_TURN_ID = tok.convert_tokens_to_ids("<end_of_turn>")
STOP_TOKEN_IDS = sorted(set(
    as_int_list(tok.eos_token_id)
    + as_int_list(model.generation_config.eos_token_id)
    + ([END_OF_TURN_ID] if END_OF_TURN_ID is not None and END_OF_TURN_ID >= 0 else [])
))

state = {"steer": [], "phase": "both", "seed": None, "temp": 0.9, "tokens": 150,
         "soft": False, "context_turns": None}
history = []


def load_dim_params(L, idx):
    with safe_open(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
        return dict(
            W_enc=f.get_slice("w_enc")[:, idx].to(device=device, dtype=torch.float32),
            W_dec=f.get_slice("w_dec")[idx].to(device=device, dtype=torch.float32),
            b_enc=f.get_slice("b_enc")[idx].to(device=device, dtype=torch.float32),
            b_dec=f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
            thr=f.get_slice("threshold")[idx].to(device=device, dtype=torch.float32),
        )


def load_inject_vec(L, F, strength):
    with safe_open(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
        return f.get_slice("w_dec")[F].to(device=device, dtype=dtype) * strength


def phase_ok(h, phase):
    is_prefill = h.shape[1] > 1
    return phase == "both" or (phase == "prefill") == is_prefill


def add_dim(arg):
    m = re.match(r"carriers\s+([\d.]+)$", arg.strip())
    if m:
        scale = float(m.group(1))
        for L, idx in CARRIERS.items():
            state["steer"].append({"kind": "dim", "layer": L, "feats": idx, "scale": scale,
                                   "params": load_dim_params(L, idx)})
        print(f"{C_SYS}carriers x{scale} @ L9/17/22/29 (16 features){C_END}")
        return
    m = re.match(r"(\d+):([\d,]+):([\d.]+)$", arg.strip())
    if not m:
        print(f"{C_ERR}usage: /dim L:F1,F2:SCALE   or   /dim carriers 0.8{C_END}")
        return
    L, feats, scale = int(m.group(1)), [int(x) for x in m.group(2).split(",")], float(m.group(3))
    if L not in SAE_LAYERS:
        print(f"{C_ERR}no SAE at layer {L}; have {sorted(SAE_LAYERS)}{C_END}")
        return
    state["steer"].append({"kind": "dim", "layer": L, "feats": feats, "scale": scale,
                           "params": load_dim_params(L, feats)})
    print(f"{C_SYS}dim f{','.join(map(str, feats))} x{scale} @ L{L}{C_END}")


def add_inject(arg):
    m = re.match(r"(\d+):(\d+):([\d.]+)$", arg.strip())
    if not m:
        print(f"{C_ERR}usage: /inject L:FEAT:STRENGTH{C_END}")
        return
    L, F, S = int(m.group(1)), int(m.group(2)), float(m.group(3))
    if L not in SAE_LAYERS:
        print(f"{C_ERR}no SAE at layer {L}; have {sorted(SAE_LAYERS)}{C_END}")
        return
    state["steer"].append({"kind": "inject", "layer": L, "feats": [F], "strength": S,
                           "vec": load_inject_vec(L, F, S)})
    print(f"{C_SYS}inject f{F} @ L{L} strength {S}{C_END}")


def show_status():
    if not state["steer"]:
        print(f"{C_SYS}no steering active{C_END}")
    for s in state["steer"]:
        if s["kind"] == "dim":
            print(f"{C_SYS}  dim    L{s['layer']} f{','.join(map(str, s['feats']))} x{s['scale']}{C_END}")
        else:
            print(f"{C_SYS}  inject L{s['layer']} f{s['feats'][0]} @ {s['strength']}{C_END}")
    print(f"{C_SYS}  phase={state['phase']} temp={state['temp']} "
          f"tokens={state['tokens']}{' soft' if state['soft'] else ''} "
          f"seed={state['seed']} turns={len(history)//2} "
          f"context={state['context_turns'] if state['context_turns'] is not None else 'all'}{C_END}")


class StopFlag(StoppingCriteria):
    """Manual interrupt; in soft mode, at the cap finish the sentence first."""

    def __init__(self, n_prompt, cap, soft):
        self.stop = False
        self.n_prompt = n_prompt
        self.cap = cap
        self.soft = soft

    def __call__(self, input_ids, scores, **kw):
        if self.stop:
            return True
        if not self.soft:
            return False               # hard mode: max_new_tokens is the cap
        n_new = input_ids.shape[1] - self.n_prompt
        if n_new < self.cap:
            return False
        tail = tok.decode(input_ids[0, -3:], skip_special_tokens=True).rstrip("*) ”\"'")
        return tail.endswith((".", "!", "?", "…", ":"))


def mk_layer_hook(ops, phase):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if not phase_ok(h, phase):
            return out
        for s in ops:
            if s["kind"] == "dim":
                p = s["params"]
                hf = h.to(torch.float32)
                pre = (hf - p["b_dec"]) @ p["W_enc"] + p["b_enc"]
                acts = pre * (pre > p["thr"])
                h = (hf + (s["scale"] - 1.0) * (acts @ p["W_dec"])).to(h.dtype)
            else:
                h = h + s["vec"]
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


def active_history():
    context_turns = state["context_turns"]
    if context_turns is None:
        return history
    # history is odd-length during generation: completed turns plus current user.
    return history[-(context_turns * 2 + 1):]


def generate():
    enc = tok.apply_chat_template(active_history(), add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    handles = []
    chunks = []
    errors = []
    try:
        steers_by_layer = {}
        for s in state["steer"]:
            steers_by_layer.setdefault(s["layer"], []).append(s)
        for L, ops in steers_by_layer.items():
            handles.append(layers[L].register_forward_hook(mk_layer_hook(ops, state["phase"])))
        if state["seed"] is not None:
            torch.manual_seed(state["seed"])
        flag = StopFlag(input_ids.shape[1], state["tokens"], state["soft"])
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                        timeout=1.0)
        # soft mode gets a grace window to finish the sentence; hard mode stops at the cap
        max_new = state["tokens"] + (80 if state["soft"] else 0)

        def _run():
            try:
                with torch.inference_mode():
                    model.generate(input_ids, attention_mask=attention_mask,
                                   max_new_tokens=max_new, do_sample=True,
                                   temperature=state["temp"], top_k=40, top_p=1.0,
                                   eos_token_id=STOP_TOKEN_IDS,
                                   pad_token_id=tok.pad_token_id,
                                   use_cache=True,
                                   streamer=streamer,
                                   stopping_criteria=StoppingCriteriaList([flag]))
            except Exception as exc:
                errors.append(exc)
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        print(f"{C_BOT}gemma>{C_END} ", end="", flush=True)
        interrupted = False
        try:
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except Empty:
                    if not thread.is_alive():
                        break
                    continue
                print(piece, end="", flush=True)
                chunks.append(piece)
        except KeyboardInterrupt:
            interrupted = True
            flag.stop = True            # Ctrl+C: stop generation, keep the chat alive
            print(f"\n{C_SYS}[interrupted — partial response kept]{C_END}", end="")
        thread.join()
        if interrupted:
            while True:                 # drain anything emitted while stopping
                try:
                    piece = next(streamer)
                except (StopIteration, Empty):
                    break
                chunks.append(piece)
        if errors:
            print(f"\n{C_ERR}[generation error: {errors[0]}]{C_END}", end="")
    finally:
        for h in handles:
            h.remove()
    print()
    return "".join(chunks)


print(f"{C_SYS}ready. /help for commands.{C_END}")
while True:
    try:
        line = input(f"{C_USER}you>{C_END} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not line:
        continue
    if line.startswith("/"):
        cmd, _, arg = line[1:].partition(" ")
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "dim":
            add_dim(arg)
        elif cmd == "boost":
            add_dim(arg)  # same hook; scale > 1 boosts
        elif cmd == "inject":
            add_inject(arg)
        elif cmd == "phase":
            if arg in ("prefill", "decode", "both"):
                state["phase"] = arg
                print(f"{C_SYS}phase={arg}{C_END}")
            else:
                print(f"{C_ERR}usage: /phase prefill|decode|both{C_END}")
        elif cmd == "status":
            show_status()
        elif cmd == "clear":
            state["steer"].clear()
            print(f"{C_SYS}steering cleared{C_END}")
        elif cmd == "reset":
            history.clear()
            print(f"{C_SYS}history cleared{C_END}")
        elif cmd == "context":
            arg = arg.strip()
            if arg == "all":
                state["context_turns"] = None
                print(f"{C_SYS}context=all{C_END}")
            else:
                try:
                    state["context_turns"] = max(0, int(arg))
                    print(f"{C_SYS}context={state['context_turns']} prior turns{C_END}")
                except ValueError:
                    print(f"{C_ERR}usage: /context N  or  /context all{C_END}")
        elif cmd == "seed":
            state["seed"] = int(arg) if arg else None
            print(f"{C_SYS}seed={state['seed']}{C_END}")
        elif cmd == "temp":
            state["temp"] = float(arg)
            print(f"{C_SYS}temp={state['temp']}{C_END}")
        elif cmd == "tokens":
            parts = arg.split()
            try:
                state["tokens"] = int(parts[0])
                state["soft"] = len(parts) > 1 and parts[1] == "soft"
                print(f"{C_SYS}tokens={state['tokens']} "
                      f"({'soft — finishes sentence' if state['soft'] else 'hard'}){C_END}")
            except (ValueError, IndexError):
                print(f"{C_ERR}usage: /tokens N  or  /tokens N soft{C_END}")
        else:
            print(f"{C_ERR}unknown command /{cmd} — /help{C_END}")
        continue
    history.append({"role": "user", "content": line})
    response = generate()
    history.append({"role": "assistant", "content": response})
