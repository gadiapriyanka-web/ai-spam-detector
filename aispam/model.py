"""Tiny dependency-free logistic regression with feature standardisation.

Why not scikit-learn? So the package runs anywhere with zero installs. The model is a
plain JSON file (weights, means, stds), so it is easy to audit, diff and swap.
"""
from __future__ import annotations

import json
import math
import random

from .features import FEATURE_NAMES, vectorize


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class TextModel:
    def __init__(self, names=None, weights=None, bias=0.0, mean=None, std=None):
        self.names = list(names or FEATURE_NAMES)
        self.w = list(weights or [0.0] * len(self.names))
        self.b = bias
        self.mean = list(mean or [0.0] * len(self.names))
        self.std = list(std or [1.0] * len(self.names))

    # ------------------------------------------------------------ inference
    def _z(self, x):
        return self.b + sum(w * (v - m) / s for w, v, m, s in zip(self.w, x, self.mean, self.std))

    def predict_vec(self, x) -> float:
        return _sigmoid(self._z(x))

    def predict(self, text: str, context: str | None = None) -> float:
        return self.predict_vec(vectorize(text, context))

    def explain(self, text: str, context: str | None = None, k: int = 3):
        """Top-k features pushing the score up (name, contribution in logit space)."""
        x = vectorize(text, context)
        contrib = [(n, w * (v - m) / s) for n, w, v, m, s in zip(self.names, self.w, x, self.mean, self.std)]
        return sorted(contrib, key=lambda t: t[1], reverse=True)[:k]

    # -------------------------------------------------------------- training
    @classmethod
    def fit(cls, X, y, epochs=400, lr=0.3, l2=1e-2, seed=0, sample_weight=None):
        rng = random.Random(seed)
        n, d = len(X), len(X[0])
        mean = [sum(r[j] for r in X) / n for j in range(d)]
        std = [(sum((r[j] - mean[j]) ** 2 for r in X) / n) ** 0.5 or 1.0 for j in range(d)]
        Z = [[(r[j] - mean[j]) / std[j] for j in range(d)] for r in X]
        sw = sample_weight or [1.0] * n
        # class balancing
        pos = sum(w for w, t in zip(sw, y) if t == 1) or 1.0
        neg = sum(w for w, t in zip(sw, y) if t == 0) or 1.0
        cw = {1: (pos + neg) / (2 * pos), 0: (pos + neg) / (2 * neg)}
        w, b = [0.0] * d, 0.0
        # Adam optimiser, full batch
        m1, v1, mb, vb = [0.0] * d, [0.0] * d, 0.0, 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        total_w = sum(sw[i] * cw[y[i]] for i in range(n))
        for t in range(1, epochs + 1):
            gw, gb = [0.0] * d, 0.0
            for zi, yi, si in zip(Z, y, sw):
                p = _sigmoid(b + sum(wj * zj for wj, zj in zip(w, zi)))
                e = (p - yi) * si * cw[yi]
                gb += e
                for j in range(d):
                    gw[j] += e * zi[j]
            gb /= total_w
            for j in range(d):
                gw[j] = gw[j] / total_w + l2 * w[j]
                m1[j] = b1 * m1[j] + (1 - b1) * gw[j]
                v1[j] = b2 * v1[j] + (1 - b2) * gw[j] ** 2
                w[j] -= lr * (m1[j] / (1 - b1 ** t)) / ((v1[j] / (1 - b2 ** t)) ** 0.5 + eps)
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb ** 2
            b -= lr * (mb / (1 - b1 ** t)) / ((vb / (1 - b2 ** t)) ** 0.5 + eps)
        return cls(FEATURE_NAMES, w, b, mean, std)

    # ----------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"names": self.names, "weights": self.w, "bias": self.b, "mean": self.mean, "std": self.std}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        if d["names"] != FEATURE_NAMES:
            raise ValueError("model was trained with a different feature set; retrain with `python -m aispam train`")
        return cls(d["names"], d["weights"], d["bias"], d["mean"], d["std"])
