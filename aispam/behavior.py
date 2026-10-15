"""Behavioral signals: does the *sender* act like an automated agent?

AI agents can write perfectly human-looking text, so timing, request
metadata and form interaction are often the more reliable tell.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict, deque

HEADLESS_UA = re.compile(
    r"headless|puppeteer|playwright|selenium|phantomjs|python-requests|httpx|aiohttp|curl/|"
    r"go-http-client|axios|node-fetch|scrapy|okhttp|langchain|autogpt|browser-?use",
    re.I,
)


def _clamp(x):
    return max(0.0, min(1.0, x))


class BehaviorTracker:
    """Keeps a sliding window of event timestamps per actor (account/IP/API key)."""

    def __init__(self, window_seconds: float = 3600, max_events: int = 200):
        self.window = window_seconds
        self.events: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_events))

    def record(self, actor: str, ts: float) -> list[float]:
        q = self.events[actor]
        q.append(ts)
        while q and ts - q[0] > self.window:
            q.popleft()
        return list(q)

    def analyze(self, actor: str, ts: float, meta: dict | None = None) -> dict:
        meta = meta or {}
        times = self.record(actor, ts)
        reasons, signals = [], {}

        # 1. Honeypot: a hidden field a human never sees.
        if meta.get("honeypot"):
            signals["honeypot"] = 1.0
            reasons.append("hidden honeypot field was filled")

        # 2. Too fast to be typed by a person.
        fill = meta.get("fill_time_s")
        if fill is not None:
            s = _clamp(1 - fill / 3.0) if fill < 3 else 0.0
            signals["fill_time"] = round(s, 3)
            if s > 0.5:
                reasons.append(f"form completed in {fill:.1f}s")

        # 3. Automation fingerprints in the user agent.
        ua = meta.get("user_agent", "")
        if ua and HEADLESS_UA.search(ua):
            signals["user_agent"] = 0.9
            reasons.append(f"automation user-agent ({HEADLESS_UA.search(ua).group(0)})")
        elif meta.get("user_agent") == "":
            signals["user_agent"] = 0.5
            reasons.append("missing user-agent")

        # 4. Rate and regularity of events.
        if len(times) >= 4:
            gaps = [b - a for a, b in zip(times, times[1:])]
            mean = statistics.mean(gaps)
            if mean > 0:
                cv = statistics.pstdev(gaps) / mean
                # Humans are bursty (cv >= ~1); cron-like agents are metronomic (cv ~ 0).
                reg = _clamp(1 - cv / 0.6) if len(gaps) >= 4 else 0
                signals["regularity"] = round(reg, 3)
                if reg > 0.6:
                    reasons.append(f"metronomic posting (gap cv={cv:.2f}, mean {mean:.1f}s)")
            span = max(times[-1] - times[0], 1.0)
            per_min = len(times) / span * 60
            rate = _clamp((per_min - 3) / 12)
            signals["rate"] = round(rate, 3)
            if rate > 0.5:
                reasons.append(f"{per_min:.0f} events/min")

        # 5. Account age: brand-new accounts that post a lot.
        age = meta.get("account_age_s")
        if age is not None and age < 3600 and len(times) >= 3:
            signals["new_account_burst"] = 0.7
            reasons.append("brand-new account posting repeatedly")

        score = 1 - _noisy_not([v for v in signals.values()])
        if signals.get("honeypot"):
            score = 1.0
        return {"score": round(_clamp(score), 3), "reasons": reasons, "signals": signals}


def _noisy_not(vals):
    p = 1.0
    for v in vals:
        p *= 1 - _clamp(v) * 0.8
    return p
