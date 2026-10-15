"""Fuses content, behavior and duplication evidence into one decision."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import text as T
from .behavior import BehaviorTracker
from .model import TextModel
from .similarity import DuplicateIndex

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "text_model.json")


@dataclass
class Config:
    review_at: float = 0.40          # risk >= this -> human review
    block_at: float = 0.70           # risk >= this -> block
    model_blend: float = 0.7         # weight of the learned text score vs. the rule score (when a model is loaded)
    behavior_weight: float = 0.90
    duplicate_weight: float = 0.90
    dup_floor: float = 0.40          # share of duplication evidence that counts with no corroboration
    cross_weight: float = 0.80       # fluent AI text + bot cadence
    judge_weight: float = 0.60       # weight of an optional LLM judge in the review band


@dataclass
class Verdict:
    risk: float
    action: str                      # allow | review | block
    ai_likelihood: float
    spam_likelihood: float
    automation_likelihood: float
    duplication: float
    text_rule_score: float = 0.0     # rule-based content evidence
    text_model_score: Optional[float] = None
    reasons: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _noisy_or(parts):
    keep = 1.0
    for p in parts:
        keep *= 1 - min(max(p, 0.0), 1.0)
    return 1 - keep


class SpamDetector:
    """Detects spam produced by LLMs or autonomous agents.

    AI-written text alone is *not* spam (people use assistants). Risk is high when AI-likeness
    co-occurs with spam intent, automation signals, or campaign-style duplication.

    Args:
        model: a `TextModel`, a path to a saved model JSON, or None for rules only.
        judge: optional callable `(text, verdict) -> float in [0,1] | None`, consulted only for
            messages in the review band (e.g. an LLM-as-judge). Keep it out of the hot path.
        trusted_actors: ids that skip behavioral scoring (e.g. your own CI bot).
        config: thresholds and fusion weights.
    """

    def __init__(self, model=None, judge: Optional[Callable] = None, trusted_actors=(), config: Optional[Config] = None,
                 review_at: Optional[float] = None, block_at: Optional[float] = None):
        self.cfg = config or Config()
        if review_at is not None:
            self.cfg.review_at = review_at
        if block_at is not None:
            self.cfg.block_at = block_at
        self.model = TextModel.load(model) if isinstance(model, (str, os.PathLike)) else model
        self.judge = judge
        self.trusted = set(trusted_actors)
        self.behavior = BehaviorTracker()
        self.dupes = DuplicateIndex()
        self._lock = threading.Lock()  # tracker/index are shared mutable state

    @classmethod
    def with_default_model(cls, **kw):
        """Use the model bundled with the package (trained on the synthetic benchmark)."""
        return cls(model=DEFAULT_MODEL_PATH, **kw)

    def score(self, text: str, actor: str = "anon", ts: Optional[float] = None,
              meta: Optional[dict] = None, context: Optional[str] = None) -> Verdict:
        c = self.cfg
        ts = time.time() if ts is None else ts
        meta = meta or {}
        t = T.analyze(text, context)           # pure, no shared state
        pm = self.model.predict(text, context) if self.model else None
        with self._lock:
            b = self.behavior.analyze(actor, ts, meta) if actor not in self.trusted else {"score": 0.0, "reasons": [], "signals": {}}
            d = self.dupes.check_and_add(actor, text)

        ai, spam, auto, dup = t["ai"], t["spam"], b["score"], d["score"]
        rule_text = _noisy_or([spam * (0.35 + 0.65 * ai), ai * 0.25])
        text_ev = rule_text if pm is None else c.model_blend * pm + (1 - c.model_blend) * rule_text

        # Identical benign text from many people is common ("thanks for sharing!"), so duplication is
        # discounted unless something else about the message or sender looks off.
        corroboration = max(spam, ai, auto, pm or 0.0)
        dup_ev = dup * (c.dup_floor + (1 - c.dup_floor) * corroboration)
        risk = _noisy_or([text_ev, auto * c.behavior_weight, dup_ev * c.duplicate_weight, ai * auto * c.cross_weight])
        reasons = t["reasons"] + b["reasons"] + d["reasons"]
        if self.model is not None and pm is not None and pm > 0.5:
            top = [n for n, v in self.model.explain(text, context, 2) if v > 0.2]
            if top:
                reasons.append("model: " + ", ".join(top))
        if meta.get("honeypot"):
            risk = 1.0

        v = Verdict(risk=risk, action="allow", ai_likelihood=ai, spam_likelihood=spam, automation_likelihood=auto,
                    duplication=dup, text_rule_score=round(rule_text, 3), text_model_score=None if pm is None else round(pm, 3),
                    reasons=reasons, signals={"text": t["signals"], "behavior": b["signals"]})
        if self.judge and c.review_at <= risk < c.block_at:
            j = self.judge(text, v)
            if j is not None:
                risk = (1 - c.judge_weight) * risk + c.judge_weight * j
                v.reasons.append(f"judge={j:.2f}")
        v.risk = round(risk, 3)
        v.action = "block" if risk >= c.block_at else "review" if risk >= c.review_at else "allow"
        return v
