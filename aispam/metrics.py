"""Evaluation metrics without numpy."""
from __future__ import annotations


def roc_auc(y, s) -> float:
    """Rank-based AUC with average ranks for ties."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    return (sum(r for r, t in zip(ranks, y) if t == 1) - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(y, s) -> float:
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    tp = fp = 0
    total_pos = sum(y)
    ap = 0.0
    prev_s = None
    for score, t in pairs:
        if t:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / total_pos if total_pos else float("nan")


def confusion(y, s, thr):
    tp = sum(1 for t, v in zip(y, s) if t == 1 and v >= thr)
    fn = sum(1 for t, v in zip(y, s) if t == 1 and v < thr)
    fp = sum(1 for t, v in zip(y, s) if t == 0 and v >= thr)
    tn = sum(1 for t, v in zip(y, s) if t == 0 and v < thr)
    return tp, fp, tn, fn


def prf(y, s, thr):
    tp, fp, tn, fn = confusion(y, s, thr)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": p, "recall": r, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_at_prevalence(recall, fpr, prevalence):
    """Precision you would see if only `prevalence` of traffic were spam."""
    denom = recall * prevalence + fpr * (1 - prevalence)
    return recall * prevalence / denom if denom else 0.0


def recall_at_fpr(y, s, max_fpr):
    """Best recall achievable while keeping FPR <= max_fpr (threshold sweep)."""
    best = 0.0
    for thr in sorted(set(s)):
        m = prf(y, s, thr)
        if m["fpr"] <= max_fpr:
            best = max(best, m["recall"])
    return best


def roc_curve(y, s):
    """(fpr, tpr) points, one per distinct threshold, from (0,0) to (1,1)."""
    pos = sum(y)
    neg = len(y) - pos
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    pts, tp, fp, i = [(0.0, 0.0)], 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        pts.append((fp / neg, tp / pos))
        i = j
    return pts


def pr_curve(y, s):
    """(recall, precision) points, one per distinct threshold."""
    pos = sum(y)
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    pts, tp, fp, i = [], 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        pts.append((tp / pos, tp / (tp + fp)))
        i = j
    return pts


def thin(points, min_step=0.004):
    """Drop points that barely move the curve, so plots stay small. Keeps both endpoints."""
    out = [points[0]]
    for p in points[1:-1]:
        if abs(p[0] - out[-1][0]) >= min_step or abs(p[1] - out[-1][1]) >= min_step:
            out.append(p)
    out.append(points[-1])
    return out


def cluster_bootstrap(y, s, groups, stat_fns, n_boot=1000, seed=0):
    """Percentile bootstrap that resamples whole groups (e.g. senders), because messages from one
    sender are not independent. `stat_fns` maps name -> f(y, s). Returns name -> (point, lo95, hi95)."""
    import random
    rng = random.Random(seed)
    idx_by = {}
    for i, g in enumerate(groups):
        idx_by.setdefault(g, []).append(i)
    keys = list(idx_by)
    point = {k: f(y, s) for k, f in stat_fns.items()}
    draws = {k: [] for k in stat_fns}
    for _ in range(n_boot):
        pick = [i for g in (rng.choice(keys) for _ in keys) for i in idx_by[g]]
        yy, ss = [y[i] for i in pick], [s[i] for i in pick]
        if sum(yy) in (0, len(yy)):
            continue
        for k, f in stat_fns.items():
            draws[k].append(f(yy, ss))
    out = {}
    for k, d in draws.items():
        d.sort()
        out[k] = (point[k], d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1])
    return out
