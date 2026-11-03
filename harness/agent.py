"""Live policies over driver observations: LAYS checkpoints (one prefill + direct heads) and the untouched-Qwen
zero-shot baseline (text actions). Both return validated schema Actions for driver.Desktop.execute.

  policy_from_spec("lays:/path/gen-001000")   trained checkpoint (trainable.safetensors from train.py)
  policy_from_spec("lays:init")               fresh heads on the untouched backbone (sanity floor)
  policy_from_spec("qwen-zeroshot")           Qwen3.5-0.8B as released, prompted for one JSON action per step
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

import objective as O
import policy as P
from convert import history_line
from driver import CAPS
from schema import FAMILIES, Action, ActionError, key_names


def sample_from(goal, o, history, prev_image):
    """Observation -> the canonical sample dict policy.render consumes (same fields the converters write)."""
    return {"goal": goal, "goal_kind": "objective", "os": getattr(o, "os", "linux"), "layout": "us", "app": o.app, "window": o.window,
            "capabilities": CAPS, "apps": o.apps, "windows": [{"id": w["id"], "title": w["title"], "app": w["app"]} for w in o.windows],
            "history": list(history[-4:]), "image": o.image, "prev_image": prev_image, "elements": o.elements} | FORMATS[FORMAT]


# Prompt-format ablation (ChatGPT review #1, 2026-10-25): P1 drops the fields no training row carries (apps / windows
# lists, window title, keyboard layout); P2 also hides the element list, like AgentNet. History effects are already
# "unknown" at inference (driver.execute), as in training.
# Without the apps / windows lists there is nothing for launch_app / focus_window to pick (25K chose launch_app on the
# first step and crashed on an empty entry list), so P1/P2 also drop those two capabilities.
FORMATS = {"P0": {}, "P1": {"apps": None, "windows": None, "window": None, "layout": None,
                            "capabilities": [c for c in CAPS if c not in ("launch_app", "focus_window")]}}
FORMATS["P2"] = FORMATS["P1"] | {"elements": None}
FORMAT = os.environ.get("LAYS_PANEL_FORMAT", "P0")

# Serving-only stop rules "LAYS+SR", off unless set and never reported as the unaided policy (done probe 2026-10-25,
# gen-010000 held-out AgentNet: on true last steps the model says done 23% with p(done) 0.27 vs 0.04 elsewhere, and
# repeats its just-executed action 26% vs 0% for humans). Per ChatGPT Reply #5: tau 0.3 is the operating point (0.2 only
# as a high-recall ablation), and a loop is never resolved by inventing another click: wait once, then fail.
STOP_TAU = float(os.environ["LAYS_STOP_TAU"]) if os.environ.get("LAYS_STOP_TAU") else None
REPEAT_GUARD = os.environ.get("LAYS_REPEAT_GUARD") == "1"
# RL rollouts (rl/rft.py): sample every head at this temperature instead of argmax; unset = greedy (all evals)
SAMPLE_T = float(os.environ["LAYS_SAMPLE_T"]) if os.environ.get("LAYS_SAMPLE_T") else None


def same_action(line, last):
    """Two history lines name the same action: same family and payload; points within 0.04 (L-inf)."""
    a, b = line.split(" ", 1) + [""], last.split(" ", 1) + [""]
    if a[0] != b[0]:
        return False
    pa, pb = (re.match(r"\(([\d.]+),([\d.]+)\)", x[1]) for x in (a, b))
    if pa and pb:
        return max(abs(float(pa[1]) - float(pb[1])), abs(float(pa[2]) - float(pb[2]))) <= 0.04
    return a[1] == b[1]


def stop_rules(model, c, s, t, history, tau=STOP_TAU, guard=REPEAT_GUARD):
    """-> (target, rule or None). Done when p(done) >= tau. When the model proposes again the click-like action it last
    executed and every screen since is unchanged: wait once for the state to settle, then fail instead of looping."""
    if tau is not None and t["family"] != "done":
        p = model.family_logits(c, O.legal_mask(s, c["h"].device)).float().softmax(-1)
        if p[FAMILIES.index("done")] >= tau:
            return {"family": "done", "route": "none"}, "stop_tau"
    acted = [i for i, h in enumerate(history) if h["action"] != "wait"]
    if not (guard and acted and t.get("route") in ("element", "pixel") and t["family"] != "drag"):
        return t, None
    since = history[acted[-1]:]
    if any(h.get("visual") != "unchanged" for h in since) or not same_action(history_line(t), since[0]["action"]):
        return t, None
    return ({"family": "fail", "route": "none"}, "repeat_fail") if len(since) > 1 else ({"family": "wait", "route": "none"}, "repeat_wait")


class Lays:
    def __init__(self, gen_dir=None, device="cuda", max_text=64, dtype=torch.bfloat16):
        self.model, self.proc = P.load(device, dtype=dtype)
        if gen_dir:
            params = dict(self.model.named_parameters())
            state = load_file(Path(gen_dir) / "trainable.safetensors")
            missing = [n for n, p in params.items() if p.requires_grad and n not in state]
            if missing:
                raise RuntimeError(f"checkpoint lacks {missing[:5]}")
            with torch.no_grad():
                for n, v in state.items():
                    params[n].copy_(v.to(params[n].device, params[n].dtype))
        self.model.eval()
        self.device, self.max_text, self.dtype = device, max_text, dtype

    @torch.no_grad()
    def generate(self, inputs):
        """Greedy payload after one prefill, retaining attention KV and DeltaNet conv/recurrent state."""
        tok, emb = self.proc.tokenizer, self.model.backbone.get_input_embeddings().weight
        end = tok.convert_tokens_to_ids(P.PAYLOAD_END)
        device = inputs["input_ids"].device
        sync = torch.get_device_module(device).synchronize
        cache = None
        if torch.device(self.device).type == "xpu":
            from transformers import StaticCache
            cache = StaticCache(config=self.model.backbone.config,
                                max_cache_len=inputs["input_ids"].shape[1] + self.max_text)
        sync(device)
        start = time.perf_counter()
        state = self.model.backbone(**inputs, past_key_values=cache, use_cache=True)
        sync(device)
        self.prefill_seconds = time.perf_counter() - start
        mask, out = inputs.get("attention_mask"), []
        position = (mask.sum(-1, keepdim=True) if mask is not None
                    else inputs["input_ids"].new_full((1, 1), inputs["input_ids"].shape[1]))
        if state.rope_deltas is not None:
            position = position + state.rope_deltas
        if cache is not None:
            full = next(layer for layer in cache.layers if hasattr(layer, "keys"))
            slots = torch.arange(full.max_cache_len, device=device)
        start = time.perf_counter()
        for _ in range(self.max_text):
            h = state.last_hidden_state[0, -1]
            logits = torch.nn.functional.linear(h.to(emb.dtype), emb)
            token = logits.argmax()
            nxt = int(token)
            if getattr(self, "trace_decisions", False):
                self.generated_token_trace.append({"token": nxt, "logprob": float(logits.float().log_softmax(-1)[nxt])})
            if nxt == end:
                break
            out.append(nxt)
            if len(out) == self.max_text:
                break
            x = {"input_ids": token.reshape(1, 1), "past_key_values": state.past_key_values,
                 "position_ids": position.unsqueeze(0).expand(3, -1, -1)}
            if cache is not None:
                # Fixed-size mask avoids HF's synchronizing dynamic mask construction on every XPU token.
                x["attention_mask"] = {"full_attention": (slots <= full.cumulative_length).view(1, 1, 1, -1),
                                       "linear_attention": None}
            elif mask is not None:
                mask = torch.cat([mask, mask.new_ones((1, 1))], 1)
                x["attention_mask"] = mask
            state = self.model.backbone(**x, use_cache=True)
            position = position + 1
        self.decode_tokens, self.decode_seconds = len(out), time.perf_counter() - start
        return tok.decode(out)

    @torch.no_grad()
    def __call__(self, task, o, history, prev, k):
        self.decode_tokens, self.decode_seconds, self.prefill_seconds = 0, 0.0, 0.0
        self.generated_token_trace = []
        s = sample_from(task.goal, o, history, prev)
        inputs, S = O.collate([s], self.proc, max_pixels=int(os.environ.get("LAYS_PIXELS", 1024 * 1024)))
        if getattr(self, "trace_decisions", False):
            self.last_prompt = P.render(S[0], self.proc.image_token)
        if torch.device(self.device).type == "xpu" and "image_grid_thw" in inputs:
            from transformers.vision_utils import get_vision_interpolation_indices_and_weights
            v = self.model.backbone.visual
            # Avoid the slow Iris Xe integer-clamp JIT; only position metadata is computed on CPU.
            indices, weights = get_vision_interpolation_indices_and_weights(
                inputs["image_grid_thw"], v.num_grid_per_side, mode=v.interpolation_mode,
                align_corners=v.interpolation_align_corners, spatial_merge_size=v.spatial_merge_size)
            inputs.update(interp_indices=indices, interp_weights=weights)
        if os.environ.get("LAYS_DUMP_PROMPTS"):  # information-parity audit: the exact serialized prompt per decision
            with open(os.environ["LAYS_DUMP_PROMPTS"], "a") as f:
                f.write(json.dumps({"task": task.id, "k": k, "prompt": P.render(S[0], "<image>")}, ensure_ascii=False) + "\n")
        inputs = {kk: v.to(self.device) for kk, v in inputs.items()}
        dtype = getattr(self, "dtype", torch.bfloat16)  # run1 wraps its training model without calling __init__
        with torch.autocast(torch.device(self.device).type, dtype=dtype, enabled=dtype != torch.float32):
            c = self.model(inputs, S)[0]
            t, self.action_head_logprob = O.decode(self.model, c, S[0], sample=SAMPLE_T is not None, temp=SAMPLE_T or 1.0)
            self.p_done = float(self.model.family_logits(c, O.legal_mask(S[0], c["h"].device)).float().softmax(-1)[FAMILIES.index("done")])
            self.last_decision_h = c["h"].detach().float().reshape(-1, c["h"].shape[-1])[-1].cpu()  # for harness probes
            if STOP_TAU is not None or REPEAT_GUARD:
                t, rule = stop_rules(self.model, c, S[0], t, history)
                if rule:  # intervention count for LAYS+SR reports
                    print(json.dumps({"sr": rule, "task": task.id, "k": k}), file=sys.stderr, flush=True)
            text = self.generate(inputs) if t.get("needs_text") else None
        if t.get("family") == "type" and t.get("text_source") in ("copy_goal", "copy_observed") and not t.get("_span_tok"):
            return Action("ask")  # chose to copy but found no valid span: escalate instead of typing nothing
        return O.to_action(t, S[0], self.proc.tokenizer, o.obs_id, text)


ZS_PROMPT = """You control a Linux desktop. Goal: {goal}
Previous actions: {hist}
Reply with exactly one JSON action and nothing else. Coordinates are 0-1000 relative to the screenshot.
{{"action":"click|double_click|right_click|hover","x":int,"y":int}}
{{"action":"type","text":str}}  {{"action":"key","keys":["ctrl","s"]}}  {{"action":"scroll","x":int,"y":int,"direction":"up|down","amount":int}}
{{"action":"drag","x":int,"y":int,"x2":int,"y2":int}}  {{"action":"done"}}  {{"action":"fail"}}"""


class QwenZeroShot:
    def __init__(self, device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        src = os.environ.get("LAYS_MODEL_DIR") or P.MODEL
        kw = {} if os.environ.get("LAYS_MODEL_DIR") else {"revision": P.REVISION}
        self.proc = AutoProcessor.from_pretrained(src, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, **kw).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, task, o, history, prev, k):
        from PIL import Image
        hist = "; ".join(h["action"] for h in history[-4:]) or "none"
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ZS_PROMPT.format(goal=task.goal, hist=hist)}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        inputs = self.proc(text=[text], images=[Image.open(o.image).convert("RGB")], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        reply = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_zero_shot(reply, o.obs_id)


def parse_zero_shot(reply, obs_id):
    """Model text -> Action. Unparseable or invalid output becomes an explicit error action (fail), never a wait."""
    try:
        return _parse(reply, obs_id).validate(set(CAPS))
    except ActionError:
        return Action("fail")


def _parse(reply, obs_id):
    m = re.search(r"\{.*?\}", reply, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        a = d.get("action")
        pt = lambda x, y: (min(max(x / 1000, 0), 1), min(max(y / 1000, 0), 1))  # noqa: E731
        if a in ("click", "double_click", "right_click", "hover"):
            return Action(a, route="pixel", point=pt(d["x"], d["y"]), obs_id=obs_id)
        if a == "type":
            return Action("type", text=str(d["text"]), text_source="generate", type_mode="insert")
        if a == "key":
            key, mods = key_names(d["keys"])
            return Action("key", key=key, mods=mods)
        if a == "scroll":
            return Action("scroll", route="pixel", point=pt(d["x"], d["y"]), scroll=(d["direction"], min(max(1, int(d.get("amount", 3))), 20)), obs_id=obs_id)
        if a == "drag":
            return Action("drag", route="pixel", point=pt(d["x"], d["y"]), dest=pt(d["x2"], d["y2"]), obs_id=obs_id)
        if a in ("done", "fail"):
            return Action(a)
    except (KeyError, TypeError, ValueError, ActionError):
        pass
    return Action("fail")


def policy_from_spec(spec):
    if spec == "qwen-zeroshot":
        return QwenZeroShot()
    if spec.startswith("lays:"):
        path = spec[5:]
        return Lays(None if path == "init" else path)
    raise ValueError(spec)
