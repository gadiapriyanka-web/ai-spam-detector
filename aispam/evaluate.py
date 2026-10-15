"""Benchmark runner: metrics, ablations, per-category breakdown, time-to-detection, latency."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict

from . import metrics as M
from .detector import Config, SpamDetector
from .model import TextModel
from .features import vectorize
from .synth import gen_stream, SPAM

CATS_ORDER = ["human_casual", "human_formal", "ai_assisted_benign", "human_praise", "human_chat_burst", "human_link",
              "classic_spam", "ai_outreach", "ai_comment", "ai_seo", "paraphrase_campaign", "agent_stealth"]


def train_model(seed=7, scale=1.0) -> TextModel:
    ev = [e for e in gen_stream("train", seed, scale) if e["text_label"] is not None]
    X = [vectorize(e["text"], e["context"]) for e in ev]
    y = [e["text_label"] for e in ev]
    return TextModel.fit(X, y)


def run_stream(det: SpamDetector, events, time_it=False):
    out, lat = [], []
    for e in events:  # events are chronological
        t0 = time.perf_counter()
        v = det.score(e["text"], e["actor"], e["ts"], e.get("meta"), e.get("context"))
        if time_it:
            lat.append((time.perf_counter() - t0) * 1000)
        out.append(v)
    return out, lat


def _row(y, s, thr):
    m = M.prf(y, s, thr)
    return {"auc": M.roc_auc(y, s), "ap": M.average_precision(y, s), "recall_at_1pct_fpr": M.recall_at_fpr(y, s, 0.01), **m}


def evaluate(model: TextModel | None, seed=7, scale=1.0, block_at=0.70, review_at=0.40) -> dict:
    ev = gen_stream("test", seed, scale)
    y = [e["label"] for e in ev]
    cats = [e["category"] for e in ev]

    rules_det = SpamDetector(model=None)
    full_det = SpamDetector(model=model)
    v_rules, _ = run_stream(rules_det, ev)
    v_full, lat = run_stream(full_det, ev, time_it=True)

    systems = {
        "Text rules only": [v.text_rule_score for v in v_rules],
        "Behavior only": [v.automation_likelihood for v in v_full],
        "Duplication only": [v.duplication for v in v_full],
        "Rules fusion (no model)": [v.risk for v in v_rules],
    }
    if model is not None:
        systems["Learned text model only"] = [v.text_model_score for v in v_full]
        systems["Full (rules + model + behavior + dup)"] = [v.risk for v in v_full]
        # leave-one-out: switch off a single layer of the full system
        for label, cfg in (("Full minus duplication", Config(duplicate_weight=0.0)),
                           ("Full minus behavior", Config(behavior_weight=0.0, cross_weight=0.0))):
            v, _ = run_stream(SpamDetector(model=model, config=cfg), ev)
            systems[label] = [x.risk for x in v]

    table = {name: _row(y, s, block_at) for name, s in systems.items()}
    final_name = "Full (rules + model + behavior + dup)" if model is not None else "Rules fusion (no model)"
    final = [v.risk for v in (v_full if model is not None else v_rules)]

    # per-category: share of events at/above block and review thresholds
    per_cat = {}
    for c in CATS_ORDER:
        idx = [i for i, k in enumerate(cats) if k == c]
        per_cat[c] = {
            "n": len(idx), "spam": c in SPAM,
            "block_rate": sum(final[i] >= block_at for i in idx) / len(idx),
            "flag_rate": sum(final[i] >= review_at for i in idx) / len(idx),
            "mean_risk": statistics.mean(final[i] for i in idx),
        }

    # time-to-detection: events an actor gets through before the first block
    by_actor = defaultdict(list)
    for i, e in enumerate(ev):
        by_actor[e["actor"]].append(i)
    ttd = defaultdict(list)
    never = defaultdict(int)
    n_actors = defaultdict(int)
    for a, idxs in by_actor.items():
        c = cats[idxs[0]]
        if c not in SPAM:
            continue
        n_actors[c] += 1
        first = next((k for k, i in enumerate(idxs) if final[i] >= block_at), None)
        if first is None:
            never[c] += 1
        else:
            ttd[c].append(first)  # 0 = blocked on first message
    detection = {c: {"actors": n_actors[c], "never_blocked": never[c],
                     "median_msgs_before_block": statistics.median(ttd[c]) if ttd[c] else None,
                     "blocked_on_first_msg": sum(1 for t in ttd[c] if t == 0) / n_actors[c]} for c in n_actors}

    # threshold sweep + precision at realistic prevalence
    sweep = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = M.prf(y, final, thr)
        sweep.append({"threshold": thr, **m,
                      "precision_at_5pct_spam": M.precision_at_prevalence(m["recall"], m["fpr"], 0.05),
                      "precision_at_1pct_spam": M.precision_at_prevalence(m["recall"], m["fpr"], 0.01)})

    lat_sorted = sorted(lat)
    perf = {"messages": len(lat), "mean_ms": statistics.mean(lat), "p50_ms": lat_sorted[len(lat) // 2],
            "p95_ms": lat_sorted[int(len(lat) * 0.95)], "p99_ms": lat_sorted[int(len(lat) * 0.99)],
            "throughput_per_s": 1000 / statistics.mean(lat)}

    return {"final_system": final_name, "events": len(ev), "spam_events": sum(y), "block_at": block_at, "review_at": review_at,
            "systems": table, "per_category": per_cat, "detection": detection, "sweep": sweep, "performance": perf}


def to_markdown(r: dict) -> str:
    L = []
    pct = lambda x: f"{100 * x:.1f}%"
    L += [f"# Benchmark results (auto-generated)\n",
          f"Held-out test stream: **{r['events']} events**, {r['spam_events']} spam ({pct(r['spam_events'] / r['events'])}). "
          f"Thresholds: review ≥ {r['review_at']}, block ≥ {r['block_at']}. Final system: **{r['final_system']}**.\n",
          "## Ablation: what each layer contributes\n",
          "| System | ROC-AUC | Avg precision | Precision | Recall | F1 | FPR | Recall @ 1% FPR |", "|---|---|---|---|---|---|---|---|"]
    for name, m in r["systems"].items():
        L.append(f"| {name} | {m['auc']:.3f} | {m['ap']:.3f} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} | {pct(m['fpr'])} | {pct(m['recall_at_1pct_fpr'])} |")
    L += ["\n_Precision/recall/F1/FPR are at the block threshold. Single-layer rows use that layer's own score at the same threshold._\n",
          "## Per-category outcomes (final system)\n", "| Category | Type | Events | Blocked | Flagged (review+block) | Mean risk |", "|---|---|---|---|---|---|"]
    for c, m in r["per_category"].items():
        L.append(f"| {c} | {'spam' if m['spam'] else 'legit'} | {m['n']} | {pct(m['block_rate'])} | {pct(m['flag_rate'])} | {m['mean_risk']:.2f} |")
    L += ["\n## Time to detection (spam senders)\n", "| Category | Senders | Blocked on first message | Median messages sent before first block | Never blocked |", "|---|---|---|---|---|"]
    for c, m in r["detection"].items():
        L.append(f"| {c} | {m['actors']} | {pct(m['blocked_on_first_msg'])} | {m['median_msgs_before_block']} | {m['never_blocked']} |")
    L += ["\n## Threshold sweep (final system)\n",
          "| Threshold | Precision | Recall | FPR | Precision if 5% of traffic is spam | Precision if 1% is spam |", "|---|---|---|---|---|---|"]
    for s in r["sweep"]:
        L.append(f"| {s['threshold']} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['fpr'])} | {pct(s['precision_at_5pct_spam'])} | {pct(s['precision_at_1pct_spam'])} |")
    if "multi_seed" in r:
        ms = r["multi_seed"]
        L += [f"\n## Stability across {len(ms['seeds'])} re-drawn test streams (seeds {ms['seeds']})\n", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|"]
        for k, label in [("auc", "ROC-AUC"), ("ap", "Avg precision"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "FPR"), ("recall_at_1pct_fpr", "Recall @ 1% FPR")]:
            m = ms[k]
            f = (lambda x: f"{100 * x:.1f}%") if k in ("precision", "recall", "fpr", "recall_at_1pct_fpr") else (lambda x: f"{x:.3f}")
            L.append(f"| {label} | {f(m['mean'])} | {f(m['std'])} | {f(m['min'])} | {f(m['max'])} |")
    p = r["performance"]
    L += ["\n## Runtime (single thread, pure Python)\n",
          f"{p['messages']} messages: mean {p['mean_ms']:.3f} ms, p50 {p['p50_ms']:.3f} ms, p95 {p['p95_ms']:.3f} ms, "
          f"p99 {p['p99_ms']:.3f} ms, about **{p['throughput_per_s']:.0f} messages/s**.\n"]
    return "\n".join(L)


def multi_seed(model, seeds=(7, 11, 13, 17, 19), block_at=0.70):
    """Re-draw the test stream with different seeds; report mean and std of headline metrics."""
    rows = []
    for sd in seeds:
        ev = gen_stream("test", sd)
        y = [e["label"] for e in ev]
        v, _ = run_stream(SpamDetector(model=model), ev)
        s = [x.risk for x in v]
        m = _row(y, s, block_at)
        rows.append(m)
    keys = ["auc", "ap", "precision", "recall", "f1", "fpr", "recall_at_1pct_fpr"]
    return {"seeds": list(seeds), **{k: {"mean": statistics.mean(r[k] for r in rows), "std": statistics.pstdev(r[k] for r in rows),
                                           "min": min(r[k] for r in rows), "max": max(r[k] for r in rows)} for k in keys}}


def main(out_json="docs/results.json", out_md="docs/RESULTS.md", model_path=None):
    model = TextModel.load(model_path) if model_path else train_model()
    r = evaluate(model)
    r["multi_seed"] = multi_seed(model)
    with open(out_json, "w") as f:
        json.dump(r, f, indent=1)
    with open(out_md, "w") as f:
        f.write(to_markdown(r))
    return r
