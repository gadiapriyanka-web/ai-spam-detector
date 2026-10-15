"""Numeric feature extraction for the learned text model."""
from __future__ import annotations

import math
import re
import statistics
import zlib

from . import text as T

FUNCTION_WORDS = frozenset(
    "the a an of to in and or but is are was were be been for with on at by from that this it as not so if "
    "we you they he she".split())
FIRST_PERSON = frozenset({"i", "i'm", "i've", "i'd", "i'll", "my", "me", "mine"})
CONTRACTION = re.compile(r"\b[a-z]+'(?:t|s|re|ve|ll|d|m)\b", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
ALLCAPS = re.compile(r"\b[A-Z]{3,}\b")

FEATURE_NAMES = [
    "log_words", "avg_word_len", "type_token", "sent_len_mean", "sent_len_cv",
    "comma_rate", "exclaim_rate", "question_rate", "emdash_rate", "digit_rate", "allcaps_rate", "emoji_rate",
    "url_count", "url_per_word", "url_risk_max", "contraction_rate", "first_person_rate", "function_word_ratio",
    "compression_ratio", "repeat_trigram_ratio",
    "sig_llm_phrases", "sig_burstiness", "sig_tics", "sig_low_specificity", "sig_evasion", "sig_promo", "sig_praise",
    "ctx_overlap", "has_context",
]


def extract(text: str, context: str | None = None) -> dict[str, float]:
    ws = T.words(text)
    n = len(ws)
    nn = max(n, 1)
    sents = [len(T.words(s)) for s in T.sentences(text)]
    sents = [x for x in sents if x]
    s_mean = statistics.mean(sents) if sents else 0.0
    s_cv = (statistics.pstdev(sents) / s_mean) if len(sents) >= 2 and s_mean else 0.0
    urls = T.URL.findall(text)
    raw = text.encode("utf-8")
    comp = len(zlib.compress(raw)) / len(raw) if len(raw) >= 40 else 1.0
    tri = [tuple(ws[i:i + 3]) for i in range(max(n - 2, 0))]
    rep3 = 1 - len(set(tri)) / len(tri) if tri else 0.0

    ctx_overlap = 0.0
    if context:
        cws = {w for w in T.words(context) if len(w) > 4}
        tws = {w for w in ws if len(w) > 4}
        ctx_overlap = len(cws & tws) / max(len(tws), 1)

    f = {
        "log_words": math.log1p(n),
        "avg_word_len": sum(map(len, ws)) / nn,
        "type_token": len(set(ws)) / nn if n >= 5 else 1.0,
        "sent_len_mean": s_mean / 30.0,
        "sent_len_cv": s_cv,
        "comma_rate": text.count(",") / nn,
        "exclaim_rate": text.count("!") / nn,
        "question_rate": text.count("?") / nn,
        "emdash_rate": (text.count("—") + text.count(" -- ")) / nn,
        "digit_rate": sum(c.isdigit() for c in text) / max(len(text), 1),
        "allcaps_rate": len(ALLCAPS.findall(text)) / nn,
        "emoji_rate": len(EMOJI.findall(text)) / nn,
        "url_count": float(min(len(urls), 3)),
        "url_per_word": len(urls) / nn,
        "url_risk_max": max((T.url_risk(u) for u in urls), default=0.0),
        "contraction_rate": len(CONTRACTION.findall(text)) / nn,
        "first_person_rate": sum(w in FIRST_PERSON for w in ws) / nn,
        "function_word_ratio": sum(w in FUNCTION_WORDS for w in ws) / nn,
        "compression_ratio": comp,
        "repeat_trigram_ratio": rep3,
        "ctx_overlap": ctx_overlap,
        "has_context": float(bool(context)),
    }
    f["sig_llm_phrases"] = T.llm_phrases(text)[0]
    f["sig_burstiness"] = T.burstiness(text)[0]
    f["sig_tics"] = T.punctuation_tics(text)[0]
    f["sig_low_specificity"] = T.low_specificity(text)[0]
    f["sig_evasion"] = T.evasion_chars(text)[0]
    f["sig_promo"] = T.promo(text)[0]
    f["sig_praise"] = T.generic_praise(text, context)[0]
    return f


def vectorize(text: str, context: str | None = None) -> list[float]:
    d = extract(text, context)
    return [d[k] for k in FEATURE_NAMES]
