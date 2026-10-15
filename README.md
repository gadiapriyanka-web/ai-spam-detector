# aispam: detect spam written by AI or AI agents

Pure Python (3.9+), no dependencies.

```bash
python3 -m aispam demo
python3 -m aispam score "text"
python3 -m aispam batch events.jsonl   # {"text","actor","ts","meta","context"}
python3 -m aispam serve --port 8080    # POST /score
python3 -m unittest discover -s tests
```

```python
from aispam import SpamDetector
det = SpamDetector()
v = det.score(text, actor="user42", ts=1.7e9,
              meta={"fill_time_s": 0.9, "user_agent": "...", "honeypot": "", "account_age_s": 120},
              context=parent_post_text)
v.action  # allow | review | block;  v.reasons explains why
```

## How it works

| Layer | Signals |
|---|---|
| Text (`text.py`) | LLM cliché phrases, uniform sentence length, stylistic tics, no concrete detail, generic praise / no topical overlap with parent, promo words and links, zero-width / mixed-script evasion |
| Behavior (`behavior.py`) | honeypot, form-fill time, automation user-agents, metronomic timing, event rate, new-account bursts |
| Duplication (`similarity.py`) | SimHash on 3-shingles plus a template "skeleton" hash, so paraphrased campaigns across accounts are caught |

`detector.py` merges them with a noisy-OR. AI-likeness alone gets low weight, because a human using an assistant is not a spammer. It counts for more alongside spam intent, bot behavior, or duplication.

## Limits and next steps

- Text-only AI detection is unreliable on short or edited text. Behavior and duplication are the stronger signals; thresholds (`review_at`, `block_at`) are untuned guesses.
- Use `review` rather than auto-block while you collect labeled data. Then fit the weights (logistic regression / gradient boosting) on your own traffic.
- Add: an LLM-as-judge for the `review` band, perplexity from a small local LM, per-IP/ASN reputation, and an LSH index for large volumes.
- Phrase lists age quickly as models change; refresh them from your own flagged data.
