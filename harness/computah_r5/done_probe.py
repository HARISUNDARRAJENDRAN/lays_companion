"""Task-completion probe on frozen r5 decision features (no r5 weights change).

Data: wrapper rollouts logged with S1_LOG_FEATURES=1 (steps.jsonl rows carry computah.h). Label for the observation at
step k = authoritative verifier success of the state r5 saw (progress after step k-1 == 1.0; at k=0 only
already_complete tasks start complete). Train-split rollouts only; CV is grouped by task family.

  python -m computah_r5.done_probe fit --runs DIR [DIR ...] --manifest M.jsonl --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def load_rows(run_roots, manifest):
    tasks = {json.loads(l)["id"]: json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()}
    X, y, groups, meta = [], [], [], []
    for root in run_roots:
        res = Path(root) / "results.jsonl"
        if not res.exists():
            continue
        for r in map(json.loads, res.read_text(encoding="utf-8").splitlines()):
            t = tasks.get(r.get("task_id"))
            steps = Path(r.get("run_dir") or "") / "steps.jsonl"
            if t is None or not steps.exists():
                continue
            rows = [json.loads(l) for l in steps.read_text(encoding="utf-8").splitlines() if l.strip()]
            done_before = t["case_kind"] == "already_complete"
            for row in rows:
                h = (row.get("computah") or {}).get("h")
                if h is not None:
                    X.append(h); y.append(1.0 if done_before else 0.0); groups.append(t["family"])
                    meta.append({"task": t["id"], "k": row.get("k"), "r5_family": (row.get("computah") or {}).get("r5_family")})
                p = row.get("oracle_progress")
                if p is not None:
                    done_before = p >= 1.0
    return torch.tensor(X), torch.tensor(y), groups, meta


def fit(X, y, l2=1e-2, epochs=400):
    mean, std = X.mean(0), X.std(0).clamp_min(1e-4)
    Z = (X - mean) / std
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.mean().clamp(1e-3, 1 - 1e-3)
    weight = torch.where(y > 0.5, 0.5 / pos, 0.5 / (1 - pos))  # class-balanced
    opt = torch.optim.LBFGS([w, b], max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y, weight=weight)
                + l2 * (w * w).sum())
        loss.backward()
        return loss
    opt.step(closure)
    return {"mean": mean, "std": std, "w": w.detach(), "b": b.detach()}


def predict(model, X):
    return torch.sigmoid(((X - model["mean"]) / model["std"]) @ model["w"] + model["b"])


def auc(p, y):
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def grouped_oof(X, y, groups, folds=5, **kw):
    fams = sorted(set(groups))
    oof = torch.zeros(len(y))
    for i in range(folds):
        held = {f for j, f in enumerate(fams) if j % folds == i}
        te = torch.tensor([g in held for g in groups])
        m = fit(X[~te], y[~te], **kw)
        oof[te] = predict(m, X[te])
    return oof


def thresholds(p, y, precision=0.95, keep_recall=0.95):
    order = torch.argsort(p, descending=True)
    tau_hi, best = 1.01, None
    tp = fp = 0
    for idx in order.tolist():
        tp += y[idx] > 0.5; fp += y[idx] < 0.5
        if tp + fp >= 5 and tp / (tp + fp) >= precision:
            tau_hi, best = float(p[idx]), (int(tp), int(fp))
    complete = torch.sort(p[y > 0.5]).values
    tau_lo = float(complete[max(0, math.floor((1 - keep_recall) * len(complete)) - 1)]) if len(complete) else 0.0
    return tau_hi, tau_lo, best


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--manifest", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--l2", type=float, default=1e-2)
    a = ap.parse_args()
    X, y, groups, meta = load_rows(a.runs, a.manifest)
    oof = grouped_oof(X, y, groups, l2=a.l2)
    tau_hi, tau_lo, hi_counts = thresholds(oof, y)
    r5_done = torch.tensor([m["r5_family"] == "done" for m in meta])
    vetoed = r5_done & (oof < tau_lo)
    report = {"n_steps": len(y), "n_complete": int(y.sum()), "families": len(set(groups)), "oof_auc": auc(oof, y),
              "tau_hi": tau_hi, "tau_hi_counts_tp_fp": hi_counts, "tau_lo": tau_lo,
              "r5_done_steps": int(r5_done.sum()), "r5_done_true": int((r5_done & (y > 0.5)).sum()),
              "veto_would_block_false_done": int((vetoed & (y < 0.5)).sum()), "veto_would_block_true_done": int((vetoed & (y > 0.5)).sum()),
              "hi_would_fire_on_incomplete": int(((oof >= tau_hi) & (y < 0.5)).sum()), "hi_would_fire_on_complete": int(((oof >= tau_hi) & (y > 0.5)).sum())}
    m = fit(X, y, l2=a.l2)
    out = {k: v.tolist() for k, v in m.items()}
    out.update(tau_hi=tau_hi, tau_lo=tau_lo, report=report, feature="r5 last_decision_h", label="verifier success of observed state")
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
