"""Collation, per-field log-likelihood terms, SFT losses and action decoding for LaysAgent.

`terms` is the single definition of log p(field | context, earlier fields) for every active field of a target.
SFT minimizes -terms (normalized per objective over the valid targets of the whole optimizer step); AWR uses
sum(terms) as log pi(a|c). Teacher forcing: argument heads are conditioned on the gold family / source patch /
start token, exactly as `decode` conditions them on its own earlier choices.

Target dict (sample["target"]); a missing key means "no label for this field" (masked, never a default):
  family, route, point|box, dest|dest_box, element, key, mods, scroll_dir, scroll_bin, text, text_source,
  span (char offsets into goal or observed text), span_in ("goal"|"observed"), type_mode, entry, effect, feasible
"""
from __future__ import annotations

import torch
from PIL import Image
from torch.nn import functional as F

from policy import (LOCATE, PAYLOAD_END, box_positives, clean, mass_nll, patch_cell_to_point, point_to_patch_cell,
                    render, softcap)
from schema import FAMILIES, KEYS, ROUTES, SCROLL_BINS, SCROLL_DIRS, TEXT_SOURCES, TYPE_MODES, Action

THUMB = 512
OBJECTIVES = ("family", "route", "patch", "cell", "element", "dest", "dest_cell", "key", "mods", "scroll_dir",
              "scroll_bin", "text_source", "type_mode", "span", "text", "entry", "effect", "feasible", "chat")


def load_images(s, budget_side=None):
    imgs = []
    if s.get("prev_image"):
        with Image.open(s["prev_image"]) as im:
            p = im.convert("RGB")
        p.thumbnail((THUMB, THUMB))
        imgs.append(p)
    with Image.open(s["image"]) as im:
        imgs.append(im.convert("RGB"))
    return imgs


def token_span(tokenizer, text, span):
    """Char span -> (start_tok, end_tok) inclusive, only when both ends fall on token boundaries of `text`
    tokenized alone (the span markers make that its in-prompt tokenization). None: not exactly copyable."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = {a: i for i, (a, b) in enumerate(enc.offset_mapping)}
    ends = {b: i for i, (a, b) in enumerate(enc.offset_mapping)}
    if span[0] in starts and span[1] in ends and starts[span[0]] <= ends[span[1]]:
        return starts[span[0]], ends[span[1]]
    return None


def prepare(s, tokenizer):
    """Resolve tokenizer-dependent labels in place: copy spans -> token indices (or fall back to generate)."""
    t = s.get("target")
    if not t or t.get("family") != "type" or "text_source" not in t:
        return s
    if t.get("text") is not None and clean(t["text"]) != t["text"]:
        t.pop("text")  # a payload containing control-token strings cannot be generated exactly: mask, never alter
    if t["text_source"] in ("copy_goal", "copy_observed"):
        src = clean(s["goal"] if t["text_source"] == "copy_goal" else s.get("observed_text") or "")
        ts = token_span(tokenizer, src, t["span"]) if "span" in t and src == (
            s["goal"] if t["text_source"] == "copy_goal" else s.get("observed_text")) else None
        if ts is None:  # visible or similar is not exact: train generation of the exact characters instead
            t["text_source"], t["_span_tok"] = "generate", None
        else:
            t["_span_tok"] = ts
    return s


def collate(samples, processor, max_pixels=1024 * 1024):
    tok = processor.tokenizer
    tok.padding_side = "right"  # causal backbone: real tokens never see right pads
    texts, images = [], []
    for s in samples:
        prepare(s, tok)
        txt = render(s, processor.image_token)
        t = s.get("target") or {}
        if t.get("family") == "type" and t.get("text_source") == "generate" and t.get("text"):
            txt += t["text"] + PAYLOAD_END  # teacher-forced payload strictly AFTER the decision token
        texts.append(txt)
        if not s.get("chat"):
            images += load_images(s)
    inputs = processor(text=texts, images=images or None, return_tensors="pt", padding=True,
                       images_kwargs={"min_pixels": 64 * 64, "max_pixels": max_pixels})
    return dict(inputs), samples


def chat_mask(ids, tokenizer):
    """Supervise assistant turns of a retention chat: tokens after '<|im_start|>assistant\\n' up to <|im_end|>."""
    start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    asst = tokenizer("assistant\n", add_special_tokens=False).input_ids
    dev, ids = ids.device, ids.tolist()  # one copy: per-token GPU reads were ~3.4K syncs per step (census 2026-10-25)
    m = [False] * len(ids)
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == start and ids[i + 1:i + 1 + len(asst)] == asst:
            j = i + 1 + len(asst)
            while j < n and ids[j] != end:
                m[j] = True
                j += 1
            if j < n:
                m[j] = True  # learn to stop
            i = j
        i += 1
    return torch.tensor(m).to(dev, non_blocking=True)


def terms(model, c, s, t, legal_families=None):
    """dict objective -> 1-D tensor of log-likelihood terms for the active fields of target t (sample s)."""
    out, dev = {}, c["h"].device
    lp = lambda z, y: z.log_softmax(-1)[y]
    fam = t.get("family")
    loc = s.get("goal_kind") == "locate"
    f = LOCATE if loc else FAMILIES.index(fam) if fam else None
    if f is None:
        return out
    if not loc:
        assert legal_families is None or legal_families[f], f"gold family {fam} is not an available capability"
        out["family"] = lp(model.family_logits(c, legal_families), f)[None]
    route = "pixel" if loc else t.get("route")
    if route is not None and not loc:
        out["route"] = lp(model.route_logits(c, f), ROUTES.index(route))[None]
    if route == "pixel" and ("point" in t or "box" in t):
        pt = t.get("point") or ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
        j, cell = point_to_patch_cell(pt, c["grid"])
        if "box" in t:
            pos = box_positives(c["grid"], t["box"], dev)
        else:
            pos = torch.zeros(c["img"].shape[0], dtype=torch.bool, device=dev)
            pos[j] = True
        out["patch"] = -mass_nll(model.patch_logits(c, f), pos)[None]
        out["cell"] = lp(model.cell_logits(c, f, j), cell)[None]
        if fam == "drag" and ("dest" in t or "dest_box" in t):
            dpt = t.get("dest") or ((t["dest_box"][0] + t["dest_box"][2]) / 2, (t["dest_box"][1] + t["dest_box"][3]) / 2)
            dj, dcell = point_to_patch_cell(dpt, c["grid"])
            dpos = box_positives(c["grid"], t["dest_box"], dev) if "dest_box" in t else \
                torch.zeros(c["img"].shape[0], dtype=torch.bool, device=dev).index_fill_(0, torch.tensor(dj, device=dev), True)
            out["dest"] = -mass_nll(model.dest_logits(c, f, j), dpos)[None]
            out["dest_cell"] = lp(model.cell_logits(c, f, dj, dest=True), dcell)[None]
    if route == "element" and "element" in t:
        out["element"] = lp(model.element_logits(c, f), t["element"])[None]
    if fam == "key" and "key" in t:
        k = KEYS.index(t["key"])
        out["key"] = lp(model.key_logits(c, f), k)[None]
        if "mods" in t:
            out["mods"] = lp(model.mod_logits(c, f, k), t["mods"])[None]
    if fam == "scroll" and "scroll_dir" in t:
        d = SCROLL_DIRS.index(t["scroll_dir"])
        out["scroll_dir"] = lp(model.scroll_logits(c, f), d)[None]
        if "scroll_bin" in t:
            out["scroll_bin"] = lp(model.bin_logits(c, f, d), SCROLL_BINS.index(t["scroll_bin"]))[None]
    if fam == "type":
        if "text_source" in t:
            out["text_source"] = lp(model.source_logits(c, f, len(c["spans"]) > 1), TEXT_SOURCES.index(t["text_source"]))[None]
        if "type_mode" in t:
            out["type_mode"] = lp(model.mode_logits(c, f), TYPE_MODES.index(t["type_mode"]))[None]
        if t.get("_span_tok") is not None:
            which = 0 if t["text_source"] == "copy_goal" else 1
            a, b = (x + 1 for x in t["_span_tok"])  # +1: index 0 is the no-valid-span key
            out["span"] = torch.stack([lp(model.start_logits(c, f, which), a), lp(model.end_logits(c, f, which, a), b)])
        if t.get("text_source") == "generate" and t.get("text"):
            ids = model._payload_ids(t["text"]).to(dev)
            out["text"] = model.text_logprobs(c, ids)
    if fam in ("launch_app", "focus_window") and "entry" in t:
        kind = "app" if fam == "launch_app" else "window"
        idx = t["entry"] + (0 if kind == "app" else len(s.get("apps") or []))
        out["entry"] = lp(model.entry_logits(c, f, len(s.get("apps") or []), kind), idx)[None]
    if "effect" in t:
        from schema import EFFECTS
        out["effect"] = lp(softcap(model.heads.effect(c["h"])), EFFECTS.index(t["effect"]))[None]
    if "feasible" in t:
        z = model.heads.feasible(c["h"])[0]
        out["feasible"] = -F.binary_cross_entropy_with_logits(z, torch.tensor(float(t["feasible"]), device=dev))[None]
    return out


def legal_mask(s, device):
    caps = s.get("capabilities")
    if not caps:
        return None
    return torch.tensor([f in caps for f in FAMILIES], device=device)


def count_targets(samples, model, tokenizer):
    """Valid target counts per objective for one optimizer step (normalizes each objective over the step)."""
    n = dict.fromkeys(OBJECTIVES, 0)
    for s in samples:
        if s.get("chat"):
            n["chat"] += 1
            continue
        prepare(s, tokenizer)
        t = s.get("target") or {}
        loc = s.get("goal_kind") == "locate"
        n["family"] += bool(t.get("family")) and not loc and s.get("goal_kind") != "chat"
        n["route"] += t.get("route") is not None and not loc
        pix = (loc or t.get("route") == "pixel") and ("point" in t or "box" in t)
        n["patch"] += pix
        n["cell"] += pix
        n["dest"] += pix and t.get("family") == "drag" and ("dest" in t or "dest_box" in t)
        n["dest_cell"] += pix and t.get("family") == "drag" and ("dest" in t or "dest_box" in t)
        n["element"] += t.get("route") == "element" and "element" in t
        n["key"] += t.get("family") == "key" and "key" in t
        n["mods"] += t.get("family") == "key" and "key" in t and "mods" in t
        n["scroll_dir"] += t.get("family") == "scroll" and "scroll_dir" in t
        n["scroll_bin"] += t.get("family") == "scroll" and "scroll_dir" in t and "scroll_bin" in t
        n["text_source"] += t.get("family") == "type" and "text_source" in t
        n["type_mode"] += t.get("family") == "type" and "type_mode" in t
        n["span"] += t.get("family") == "type" and t.get("_span_tok") is not None
        n["text"] += t.get("family") == "type" and t.get("text_source") == "generate" and bool(t.get("text"))
        n["entry"] += t.get("family") in ("launch_app", "focus_window") and "entry" in t
        n["effect"] += "effect" in t
        n["feasible"] += "feasible" in t
    return n


def step_loss(model, ctx, samples, inputs, norm, weights=None):
    """Sum over samples of -terms / norm[objective]: each objective is the mean over its valid targets in the
    optimizer step (norm counts the whole step, so accumulation over microbatches gives the same gradient)."""
    weights = weights or {}
    total, parts = 0.0, {}
    tok = model._tokenizer
    for b, (c, s) in enumerate(zip(ctx, samples)):
        if s.get("chat"):
            m = chat_mask(inputs["input_ids"][b], tok)
            lpc = model.chat_logprobs(c, m)
            v = -lpc.mean() / max(norm["chat"], 1)
            total = total + weights.get("chat", 1.0) * v
            parts["chat"] = parts.get("chat", 0.0) + v.detach().double()  # stays on the GPU: float() here was a sync
            continue
        legal = legal_mask(s, c["h"].device)
        for k, v in terms(model, c, s, s.get("target") or {}, legal).items():
            loss = -v.mean() / max(norm[k], 1)  # span (2 terms) and text (n tokens) count once per target
            total = total + weights.get(k, 1.0) * loss
            parts[k] = parts.get(k, 0.0) + loss.detach().double()  # float64 sums, as the Python floats were
    return total, parts


# -------------------------------------------------------------------- decoding / sampling
@torch.no_grad()
def decode(model, c, s, sample=False, generator=None, temp=1.0):
    """Choose a full action by the same factorization `terms` scores. Returns (target dict, log p). Text
    generation for text_source=generate is left to the caller (serve.generate) and marked `needs_text`."""
    pick = (lambda z: int(torch.multinomial((z.float() / temp).softmax(-1), 1, generator=generator))) if sample else (lambda z: int(z.argmax()))
    lp = lambda z, y: float(z.log_softmax(-1)[y])
    t, total = {}, 0.0
    zf = model.family_logits(c, legal_mask(s, c["h"].device))
    f = pick(zf)
    total += lp(zf, f)
    fam = FAMILIES[f]
    t["family"] = fam
    if fam in ("click", "double_click", "right_click", "hover", "drag", "type", "scroll"):
        zr = model.route_logits(c, f)
        r = pick(zr)
        total += lp(zr, r)
        t["route"] = ROUTES[r]
        if t["route"] == "pixel":
            zp = model.patch_logits(c, f)
            j = pick(zp)
            zc = model.cell_logits(c, f, j)
            cell = pick(zc)
            total += lp(zp, j) + lp(zc, cell)
            t["point"] = patch_cell_to_point(j, cell, c["grid"])
            if fam == "drag":
                zd = model.dest_logits(c, f, j)
                dj = pick(zd)
                zdc = model.cell_logits(c, f, dj, dest=True)
                dc = pick(zdc)
                total += lp(zd, dj) + lp(zdc, dc)
                t["dest"] = patch_cell_to_point(dj, dc, c["grid"])
        elif t["route"] == "element":
            ze = model.element_logits(c, f)
            e = pick(ze)
            total += lp(ze, e)
            t["element"] = e
    if fam == "key":
        zk = model.key_logits(c, f)
        k = pick(zk)
        zm = model.mod_logits(c, f, k)
        m = pick(zm)
        total += lp(zk, k) + lp(zm, m)
        t["key"], t["mods"] = KEYS[k], m
    if fam == "scroll":
        zd = model.scroll_logits(c, f)
        d = pick(zd)
        zb = model.bin_logits(c, f, d)
        bn = pick(zb)
        total += lp(zd, d) + lp(zb, bn)
        t["scroll_dir"], t["scroll_bin"] = SCROLL_DIRS[d], SCROLL_BINS[bn]
    if fam == "type":
        zs = model.source_logits(c, f, len(c["spans"]) > 1)
        src = pick(zs)
        zm = model.mode_logits(c, f)
        mode = pick(zm)
        total += lp(zs, src) + lp(zm, mode)
        t["text_source"], t["type_mode"] = TEXT_SOURCES[src], TYPE_MODES[mode]
        if t["text_source"] in ("copy_goal", "copy_observed"):
            which = 0 if t["text_source"] == "copy_goal" else 1
            za = model.start_logits(c, f, which)
            a = pick(za)
            zb = model.end_logits(c, f, which, a)
            b = pick(zb)
            total += lp(za, a) + lp(zb, b)
            t["_span_tok"] = None if a == 0 or b == 0 else (a - 1, b - 1)  # 0 = no valid span -> caller asks
        elif t["text_source"] == "generate":
            t["needs_text"] = True
    if fam in ("launch_app", "focus_window"):
        kind = "app" if fam == "launch_app" else "window"
        n_apps = len(s.get("apps") or [])
        ze = model.entry_logits(c, f, n_apps, kind)
        e = pick(ze)
        total += lp(ze, e)
        t["entry"] = e if kind == "app" else e - n_apps
    return t, total


def to_action(t, s, tokenizer, obs_id, generated_text=None):
    """Decoded target -> validated schema Action (raises ActionError; never an implicit wait)."""
    fam = t["family"]
    a = Action(fam, route=t.get("route", "none"), obs_id=obs_id if t.get("route", "none") != "none" else None)
    if a.route == "pixel":
        a.point = tuple(t["point"])
    if a.route == "element":
        a.element = t["element"]
    if fam == "drag":
        a.dest = tuple(t["dest"])
    if fam == "key":
        a.key, a.mods = t["key"], t["mods"]
    if fam == "scroll":
        a.scroll = (t["scroll_dir"], t["scroll_bin"])
    if fam == "type":
        a.text_source, a.type_mode = t["text_source"], t["type_mode"]
        if t["text_source"] in ("copy_goal", "copy_observed") and t.get("_span_tok"):
            src = s["goal"] if t["text_source"] == "copy_goal" else s["observed_text"]
            off = tokenizer(clean(src), add_special_tokens=False, return_offsets_mapping=True).offset_mapping
            i, j = t["_span_tok"]
            a.span = (off[i][0], off[j][1])
            a.text = clean(src)[a.span[0]:a.span[1]]
        elif t["text_source"] == "generate":
            a.text = generated_text
    if fam == "launch_app":
        a.app_id = s["apps"][t["entry"]]["id"]
    if fam == "focus_window":
        a.window_id = s["windows"][t["entry"]]["id"]
    return a.validate(set(s["capabilities"]) if s.get("capabilities") else None)
