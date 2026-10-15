"""Combines content, behavior and duplication evidence into one decision."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import text as T
from .behavior import BehaviorTracker
from .similarity import DuplicateIndex


@dataclass
class Verdict:
    risk: float
    action: str                      # allow | review | block
    ai_likelihood: float
    spam_likelihood: float
    automation_likelihood: float
    duplication: float
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SpamDetector:
    """Detects spam produced by LLMs or autonomous agents.

    Key idea: AI-written text alone is *not* spam (people use assistants). Risk is high when
    AI-likeness co-occurs with spam intent, automation signals, or campaign-style duplication.
    """

    def __init__(self, review_at: float = 0.40, block_at: float = 0.70):
        self.review_at, self.block_at = review_at, block_at
        self.behavior = BehaviorTracker()
        self.dupes = DuplicateIndex()

    def score(self, text: str, actor: str = "anon", ts: float | None = None,
              meta: dict | None = None, context: str | None = None) -> Verdict:
        ts = time.time() if ts is None else ts
        t = T.analyze(text, context)
        b = self.behavior.analyze(actor, ts, meta)
        d = self.dupes.check_and_add(actor, text)

        ai, spam, auto, dup = t["ai"], t["spam"], b["score"], d["score"]

        # Noisy-OR over independent lines of evidence.
        parts = [
            spam * (0.35 + 0.65 * ai),   # spammy content, more damning if machine-written
            auto * 0.90,                 # bot-like behavior
            dup * 0.90,                  # campaign / template reuse
            ai * 0.25,                   # AI text alone: weak evidence
            # Agent writing fluent text on a bot-like cadence even if nothing else fires
            ai * auto * 0.8,
        ]
        keep = 1.0
        for p in parts:
            keep *= 1 - min(max(p, 0.0), 1.0)
        risk = 1 - keep
        if (meta or {}).get("honeypot"):
            risk = 1.0

        action = "block" if risk >= self.block_at else "review" if risk >= self.review_at else "allow"
        return Verdict(
            risk=round(risk, 3), action=action,
            ai_likelihood=ai, spam_likelihood=spam,
            automation_likelihood=auto, duplication=dup,
            reasons=t["reasons"] + b["reasons"] + d["reasons"],
            signals={"text": t["signals"], "behavior": b["signals"]},
        )
