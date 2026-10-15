"""Content signals: is the text machine-written, and does it look like spam?

Every signal returns (score in 0..1, human-readable reason or None).
These are heuristics, not proof; combine them with behavior and duplication.
"""
from __future__ import annotations

import math
import re
import statistics
import unicodedata

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z']+")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)

# Phrases that LLM output over-uses, and that humans rarely use in short comments/messages.
LLM_PHRASES = [
    "as an ai", "i hope this message finds you well", "i hope this finds you well",
    "delve into", "delving into", "in today's fast-paced", "in the ever-evolving",
    "ever-evolving landscape", "rich tapestry", "testament to", "it's important to note",
    "it is important to note", "navigate the complexities", "unlock the full potential",
    "game-changer", "game changer", "seamless", "elevate your", "leverage",
    "in conclusion", "overall,", "furthermore", "moreover", "i'd be happy to",
    "feel free to reach out", "don't hesitate to", "i came across your", "i was impressed by",
    "your insightful", "this resonates", "great insights", "thanks for sharing this valuable",
    "not just", "plays a crucial role", "a myriad of", "multifaceted", "holistic approach",
    "cutting-edge", "take it to the next level", "unleash", "dive into", "let's dive",
    "i apologize for any confusion", "certainly!", "absolutely!",
]

GENERIC_PRAISE = [
    "great post", "great article", "nice post", "very informative", "well written",
    "thanks for sharing", "love this", "so true", "amazing content", "keep up the great work",
    "this is so helpful", "valuable insights", "insightful post",
]

PROMO_WORDS = [
    "buy now", "limited time", "click here", "free trial", "discount", "guaranteed",
    "earn money", "work from home", "crypto", "casino", "seo services", "backlinks",
    "dm me", "whatsapp", "telegram", "visit our", "check out my", "book a call",
    "boost your", "increase your traffic", "special offer", "act now", "100%",
    "book a demo", "schedule a demo", "free demo", "request a demo", "grow revenue", "revenue 3x",
    "working from home", "per day", "make money", "claim your", "you won", "free gift", "airdrop", "winner",
]

TRUSTED_DOMAINS = (
    "github.com", "gitlab.com", "stackoverflow.com", "wikipedia.org", "arxiv.org", "python.org", "readthedocs.io",
    "kubernetes.io", "postgresql.org", "mozilla.org", "mit.edu", "nih.gov", "rust-lang.org", "npmjs.com", "pypi.org",
)
CHEAP_TLDS = {"biz", "xyz", "online", "top", "click", "icu", "live", "site", "shop", "club", "work", "buzz", "info"}
PROMO_PATH = {"offer", "deal", "promo", "join", "start", "go", "ref", "buy", "free", "claim", "signup"}


def url_risk(url: str) -> float:
    """0 (reputable) .. 1 (looks like a throwaway marketing link). Heuristic only: no network lookups."""
    from urllib.parse import urlparse
    u = urlparse(url if "//" in url else "//" + url)
    host = (u.hostname or "").lower()
    if not host:
        return 0.3
    if any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS) or host.endswith((".edu", ".gov")):
        return 0.0
    labels = host.split(".")
    tld, name = labels[-1], labels[-2] if len(labels) >= 2 else host
    risk = 0.25
    if tld in CHEAP_TLDS:
        risk += 0.3
    if "-" in name:
        risk += 0.25
    if any(seg in PROMO_PATH for seg in u.path.lower().strip("/").split("/")):
        risk += 0.2
    return min(risk, 1.0)


# Characters that are used to evade filters
ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid_ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear between."""
    if hi == lo:
        return float(x >= hi)
    return _clamp((x - lo) / (hi - lo))


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


# ---------------------------------------------------------------- AI-likeness

def llm_phrases(text: str):
    low = text.lower()
    hits = [p for p in LLM_PHRASES if p in low]
    n_words = max(len(words(text)), 1)
    density = len(hits) / (n_words / 50)  # hits per 50 words
    score = _sigmoid_ramp(density, 0.3, 1.5)
    return score, (f"LLM-typical phrasing: {', '.join(hits[:4])}" if hits and score > 0.2 else None)


def burstiness(text: str):
    """Humans vary sentence length a lot; LLM prose is unusually even."""
    lens = [len(words(s)) for s in sentences(text)]
    lens = [n for n in lens if n > 0]
    if len(lens) < 4:
        return 0.0, None
    mean = statistics.mean(lens)
    cv = statistics.pstdev(lens) / mean if mean else 0
    score = 1 - _sigmoid_ramp(cv, 0.25, 0.65)
    return score * 0.8, (f"unusually uniform sentence lengths (cv={cv:.2f})" if score > 0.5 else None)


def punctuation_tics(text: str):
    n = max(len(words(text)), 1)
    em = text.count("—") + text.count(" -- ")
    contrast = len(re.findall(r"\bnot (?:just|only|merely)\b.{0,60}\bbut\b", text.lower()))
    tricolon = len(re.findall(r"\b\w+, \w+,? and \w+\b", text.lower()))
    emoji_lead = bool(re.match(r"^\W{0,3}[\U0001F300-\U0001FAFF☀-➿]", text))
    raw = (em / n) * 40 + contrast * 0.6 + (tricolon / max(len(sentences(text)), 1)) * 0.5 + emoji_lead * 0.2
    score = _sigmoid_ramp(raw, 0.3, 1.5)
    return score * 0.7, ("stylistic tics (em-dashes, 'not just X but Y', lists of three)" if score > 0.4 else None)


def low_specificity(text: str):
    """Long text that names nothing concrete: no numbers, no proper nouns, no quotes."""
    ws = words(text)
    if len(ws) < 25:
        return 0.0, None
    digits = len(re.findall(r"\d", text))
    # capitalised words that are not sentence-initial
    proper = len(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", text))
    concreteness = digits / len(ws) * 5 + proper / len(ws) * 3
    score = 1 - _sigmoid_ramp(concreteness, 0.0, 0.5)
    return score * 0.6, ("no concrete details (no numbers, names, or specifics)" if score > 0.8 else None)


def evasion_chars(text: str):
    zw = sum(ch in ZERO_WIDTH for ch in text)
    mixed = 0
    for tok in WORD.findall(text) + re.findall(r"\w+", text):
        scripts = {unicodedata.name(c, "?").split()[0] for c in tok if c.isalpha()}
        if {"LATIN", "CYRILLIC"} <= scripts or {"LATIN", "GREEK"} <= scripts:
            mixed += 1
    if zw or mixed:
        return 1.0, f"filter-evasion characters (zero-width={zw}, mixed-script words={mixed})"
    return 0.0, None


# ------------------------------------------------------------------ Spam-ness

def promo(text: str):
    low = text.lower()
    hits = [p for p in PROMO_WORDS if p in low]
    urls = URL.findall(text)
    risks = [url_risk(u) for u in urls]
    n = max(len(words(text)), 1)
    raw = len(hits) * 0.4 + sum(risks) * 0.8 + (len(urls) / n) * 3 * max(risks, default=0)
    score = _sigmoid_ramp(raw, 0.2, 1.2)
    bits = []
    if hits:
        bits.append("promo wording: " + ", ".join(hits[:3]))
    if urls:
        bits.append(f"{len(urls)} link(s), max link risk {max(risks):.2f}")
    return score, ("; ".join(bits) if score > 0.3 else None)


def generic_praise(text: str, context: str | None = None):
    """Praise that could be pasted under any post. If we know what it replies to,
    check whether it actually engages with that content."""
    low = text.lower()
    praise = [p for p in GENERIC_PRAISE if p in low]
    ws = set(words(text))
    overlap = None
    if context:
        cws = {w for w in words(context) if len(w) > 4}
        content_ws = {w for w in ws if len(w) > 4}
        overlap = len(cws & content_ws) / max(len(content_ws), 1)
    score = 0.0
    if praise:
        score = 0.6
        if len(ws) < 40:
            score += 0.2
    if overlap is not None and len(ws) > 12:
        score = max(score, (1 - _sigmoid_ramp(overlap, 0.02, 0.2)) * 0.7)
    reason = None
    if score > 0.4:
        reason = "generic engagement" + (f" ({overlap:.0%} topical overlap with parent)" if overlap is not None else "")
    return _clamp(score), reason


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in freq.values())


AI_SIGNALS = {
    "llm_phrases": (llm_phrases, 0.35),
    "burstiness": (burstiness, 0.20),
    "punctuation_tics": (punctuation_tics, 0.15),
    "low_specificity": (low_specificity, 0.20),
    "evasion_chars": (evasion_chars, 0.10),
}


def analyze(text: str, context: str | None = None) -> dict:
    """Return {'ai': score, 'spam': score, 'reasons': [...], 'signals': {...}}."""
    signals, reasons = {}, []
    ai_total = 0.0
    for name, (fn, w) in AI_SIGNALS.items():
        s, why = fn(text)
        signals[name] = round(s, 3)
        ai_total += s * w
        if why:
            reasons.append(why)
    # Few strong signals matter more than many weak ones: boost by the max signal.
    ai = _clamp(0.6 * ai_total / sum(w for _, w in AI_SIGNALS.values()) * 1.8 + 0.4 * max(signals.values()))

    p, why = promo(text)
    g, why2 = generic_praise(text, context)
    signals["promo"], signals["generic_praise"] = round(p, 3), round(g, 3)
    reasons += [r for r in (why, why2) if r]
    spam = _clamp(max(p, g, signals["evasion_chars"]))
    # Short texts give the AI heuristics too little to work with.
    if len(words(text)) < 12:
        ai *= 0.5
    return {"ai": round(ai, 3), "spam": round(spam, 3), "reasons": reasons, "signals": signals}
