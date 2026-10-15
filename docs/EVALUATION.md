# Evaluation

Reproduce everything below with:

```bash
python3 -m aispam bench                  # regenerates docs/RESULTS.md and docs/results.json (about 5 s)
python3 -m unittest discover -s tests    # includes benchmark regression gates
```

## 1. What was measured, and how

**Data.** `aispam/synth.py` generates a chronologically interleaved event stream of messages with actor ids,
timestamps, request metadata and optional parent-post context. There are 12 scenario families:

| Family | Label | What it models |
|---|---|---|
| `human_casual` | legit | short specific replies, heavy-tailed human timing |
| `human_formal` | legit | long, polite, detailed ("Furthermore", names, numbers) |
| `ai_assisted_benign` | legit (hard) | LLM-polished but specific, no promo, human cadence |
| `human_praise` | legit (hard) | "Great post!" from real people, half generic |
| `human_chat_burst` | legit (hard) | fast, irregular chat replies |
| `human_link` | legit (hard) | helpful GitHub/docs/StackOverflow links |
| `ai_outreach` | spam | LLM sales DMs, form-fill bots, repeat senders |
| `ai_comment` | spam | generic AI praise plus a plug link, with parent context |
| `ai_seo` | spam | SEO/backlink pitches |
| `classic_spam` | spam | human-style "BUY CHEAP!!!" spam |
| `paraphrase_campaign` | spam | one template reworded across many one-shot accounts |
| `agent_stealth` | spam | fluent, on-topic, non-promotional text; only cadence and metadata give it away (half have jittered timing, 45% spoof a browser UA) |

**Protocol.**
- Every phrase pool is split by index parity: the *train* split uses even entries, the *test* split uses odd
  entries. The learned model never sees test wording.
- The text model trains only on the train stream. Test streams use unseen seeds; headline numbers are the mean
  of 5 re-drawn test streams.
- Events are scored **in chronological order by one detector instance**, so behavior and duplication layers see
  realistic history (an agent is only recognised as metronomic after a few messages).
- Metrics: ROC-AUC and average precision (threshold-free), precision/recall/F1/FPR at the block threshold 0.70,
  recall at fixed 1% FPR, and precision at realistic spam prevalence.

## 2. Headline results

Mean over 5 re-drawn test streams (block threshold 0.70):

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ROC-AUC | 0.998 | 0.001 | 0.997 | 0.999 |
| Precision | 99.0% | 0.4% | 98.5% | 99.5% |
| Recall | 87.4% | 2.0% | 85.0% | 90.5% |
| F1 | 0.928 | 0.013 | 0.913 | 0.948 |
| False-positive rate | 0.5% | 0.2% | 0.3% | 0.8% |
| Recall at 1% FPR | 92.3% | 2.0% | 90.6% | 95.7% |

Throughput: mean 1.0 ms per message (p95 1.8 ms, p99 2.2 ms), about 985 messages/s on one thread.

## 3. What each layer contributes (ablation, seed 7, 2,241 events)

| System | ROC-AUC | Recall at block | FPR at block | Recall at 1% FPR |
|---|---|---|---|---|
| Text rules only | 0.789 | 23.8% | 0.0% | 61.2% |
| Learned text model only | 0.797 | 59.9% | 0.0% | 63.9% |
| Behavior only | 0.814 | 45.7% | 0.0% | 61.4% |
| Duplication only | 0.589 | 10.2% | 9.6% | 0.0% |
| Rules fusion (no model) | 0.977 | 66.6% | 0.6% | 73.9% |
| **Full system** | **0.997** | **88.7%** | **0.4%** | **93.5%** |
| Full minus behavior | 0.965 | 60.2% | 0.4% | 69.0% |
| Full minus duplication | 1.000 | 83.4% | 0.0% | 99.5% |

Reading it:

1. **No single layer is enough** (single-layer AUC is 0.59 to 0.81). Fusion reaches 0.997, so the layers cover
   different spam. The single-layer rows are capped by design: the 289 stealth-agent events have
   normal-looking text, so no text-only system can exceed roughly 0.8 AUC here.
2. **Behavior is the most valuable layer** against fluent agent spam: removing it costs about 28 points of
   block-threshold recall.
3. **The learned model beats the hand rules on recall** at the same threshold (59.9% vs 23.8% text-only) and
   adds about 22 points recall to the fusion, with the same FPR.
4. **Duplication is the weakest layer, and on this benchmark it is net-ambiguous.** It raises recall at the
   block threshold (83.4% to 88.7%) but removing it *improves* ranking (AUC 1.000, recall at 1% FPR 99.5%),
   because the synthetic legitimate traffic reuses a small set of templates and so looks more repetitive than
   real traffic does. Whether it earns its place depends on your data; measure it with
   `Config(duplicate_weight=0)`.

## 4. Per-category outcomes (seed 7)

| Category | Type | Blocked | Flagged (review or block) |
|---|---|---|---|
| human_casual | legit | 0.0% | 0.0% |
| human_formal | legit | 0.0% | 0.0% |
| human_praise | legit | 0.0% | 0.0% |
| human_link | legit | 0.0% | 1.7% |
| human_chat_burst | legit | 0.0% | 2.9% |
| ai_assisted_benign | legit | 2.8% | 16.6% |
| ai_outreach | spam | 100% | 100% |
| ai_comment | spam | 100% | 100% |
| ai_seo | spam | 100% | 100% |
| classic_spam | spam | 85.8% | 99.2% |
| agent_stealth | spam | 84.8% | 96.2% |
| paraphrase_campaign | spam | 72.5% | 100% |

**Time to detection.** 80% of stealth agents are blocked on their *first* message, from metadata alone (fast
form fill, automation UA). The rest are caught after a few messages once cadence evidence accumulates. 33 of
the 120 one-shot campaign accounts and 17 of the 120 classic spammers are never blocked, though most are
flagged for review.

## 5. Operating point and base rates

Precision depends on how much traffic is spam. The benchmark is 37% spam, which is far higher than most
products see. Using the measured recall and FPR, expected precision at the block threshold:

| Threshold | Recall | FPR | Precision if 5% spam | Precision if 1% spam |
|---|---|---|---|---|
| 0.40 | 98.6% | 2.6% | 66.4% | 27.5% |
| 0.50 | 94.6% | 1.3% | 79.6% | 42.8% |
| 0.70 (default) | 88.7% | 0.4% | 92.9% | 71.6% |
| 0.80 | 76.0% | 0.0% | 100% | 100% |

Even a 0.4% FPR matters when spam is rare: at 1% prevalence, about 3 in 10 blocks would hit a real user (this
excludes any FPR that unseen real-world legit behavior would add). This is why the default routes the
0.40 to 0.70 band to review rather than blocking, and why you should choose the threshold from *your* costs.

## 6. Bugs this benchmark caught (and what was changed)

| Finding | Cause | Fix |
|---|---|---|
| About 49% of casual human messages blocked | duplication layer treated naturally repeated short phrases as campaigns | length-scaled, sender-count-based duplication score; corroboration discount |
| Legit GitHub-link message blocked by the bundled model | model had learned "has a URL = spam" because every synthetic spam sample contained a link | added `human_link` hard negatives; added `url_risk` (domain/TLD/path reputation) as a feature and in `promo` |
| "Earn $40 per day working from home!!!" missed | promo lexicon lacked these stems | broadened lexicon |

Two of these fixes were made *after* looking at test-stream results (the benchmark does not have a separate
validation split). Treat the post-fix numbers as optimistic by that amount; the five-seed re-draws share
wording with the test pools. The regression gates in `tests/test_benchmark.py` use a different seed and
thresholds set slightly below current performance, so they act as tripwires and not as accuracy claims.

## 7. Known limitations

- **Synthetic data, single author.** Templates and detector were written together. The result of 0.998 AUC
  is an upper bound on what to expect, not a prediction. The most valuable next step is labeled real traffic.
- **Shortcut risk in the learned model.** Top features are link risk, promo score and digit rate, which
  reflect how the synthetic spam was written. A model trained on real data will weight differently.
- **Adaptive adversaries.** A stealth agent that randomises timing widely, spoofs a browser UA, fills slowly
  and uses distinct non-promotional text per message is effectively invisible to this system. The tests cover
  cheap evasions (zero-width padding, upper-casing, homoglyphs) but not an adversary who reads the source.
- **Paraphrase at scale.** SimHash on short pitches often misses heavy reworking; in the demo, two reworded
  pitches were not recognised as duplicates (each was caught on text and link signals instead).
- **Legit automation** (CI bots, integrations) looks like an agent; use `trusted_actors`.
- **Non-English text.** The lexicons and function-word features are English. Behavior and duplication layers
  are language-independent.
- **AI-assisted benign users** are the hardest legit group: 16.6% are flagged for review (2.8% blocked).
- **Latency numbers** are for one thread in CPython 3.9 on a laptop, with an in-memory index.

## 8. How to evaluate on your own data

```bash
python3 -m aispam gen --split test > synthetic.jsonl     # see the expected schema
python3 -m aispam batch your_events.jsonl --model > scored.jsonl
```

Each input line: `{"id", "text", "actor", "ts", "meta", "context"}`. Label a few hundred events (including
borderline ones and your real legit power users), then compute AUC, recall at your acceptable FPR, and the
per-segment error rates with `aispam.metrics`. [INTEGRATION.md](INTEGRATION.md) covers retraining the model.
