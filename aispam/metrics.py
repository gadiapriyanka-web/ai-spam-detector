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
