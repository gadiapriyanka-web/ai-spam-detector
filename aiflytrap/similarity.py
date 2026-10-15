"""Near-duplicate and template detection across senders.

LLM-driven spam campaigns rarely repeat text verbatim; they paraphrase one
template. Two fingerprints catch that:
  * SimHash over word 3-shingles  -> light paraphrases
  * "skeleton" hash               -> same template with names/numbers/links swapped
"""
from __future__ import annotations

import hashlib
import re
from collections import deque

WORD = re.compile(r"[a-z0-9']+")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def simhash(text: str, k: int = 3) -> int:
    toks = WORD.findall(text.lower())
    shingles = [" ".join(toks[i:i + k]) for i in range(max(len(toks) - k + 1, 1))]
    v = [0] * 64
    for sh in shingles:
        h = _h64(sh)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(64) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def skeleton(text: str) -> int:
    """Mask the parts a template fills in (links, numbers, capitalised names)."""
    t = URL.sub("<URL>", text)
    t = re.sub(r"\d+", "<N>", t)
    t = re.sub(r"\b[A-Z][a-z]+\b", "<W>", t)
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return _h64(t)


class DuplicateIndex:
    """Sliding window of recent messages. Fine for thousands per window; swap for
    an LSH index / vector DB at larger scale."""

    def __init__(self, capacity: int = 5000, max_distance: int = 8, min_words: int = 8):
        self.items: deque = deque(maxlen=capacity)  # (sender, simhash, skeleton)
        self.max_distance = max_distance
        self.min_words = min_words

    def check_and_add(self, sender: str, text: str) -> dict:
        n_words = len(WORD.findall(text.lower()))
        if n_words < self.min_words:
            return {"score": 0.0, "reasons": [], "matches": 0, "senders": 0}
        sh, sk = simhash(text), skeleton(text)
        near, same_skel, senders = 0, 0, set()
        for other, osh, osk in self.items:
            if other == sender and osh == sh:
                continue  # an exact self-repeat is handled by the behavior side
            if hamming(sh, osh) <= self.max_distance:
                near += 1
                senders.add(other)
            elif osk == sk:
                same_skel += 1
                senders.add(other)
        self.items.append((sender, sh, sk))
        matches = near + same_skel
        # Many *distinct* senders using one template is a campaign; long templates are rarer by
        # chance than short ones ("thanks for sharing!"), so evidence scales with length.
        length_factor = min(1.0, max(0.0, (n_words - 6) / 14))
        score = (1 - 0.55 ** min(len(senders), 5)) * length_factor
        reasons = []
        if matches and score > 0.05:
            reasons.append(f"near-duplicate of {matches} recent message(s) from {len(senders)} sender(s)")
        return {"score": round(score, 3), "reasons": reasons, "matches": matches, "senders": len(senders)}
