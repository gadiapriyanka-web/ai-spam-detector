# Integration guide

## 1. Choose where to call it

Call `SpamDetector.score()` **synchronously on write paths**: comment/DM/post creation, signup, contact
forms, review submission. At about 1 ms per message it fits inside a request. Keep **one detector instance per
process** (it owns the per-actor history and the duplicate window) and reuse it. For several processes, either
route each actor consistently to one process (sticky key), or run the HTTP service as a single shared scorer.

```python
det = SpamDetector.with_default_model(trusted_actors={"ci-bot"})

def on_comment(req):
    v = det.score(req.body, actor=req.account_id or req.ip, ts=req.time,
                  meta={"fill_time_s": req.form_open_to_submit, "user_agent": req.headers.get("User-Agent", ""),
                        "account_age_s": req.account_age, "honeypot": req.form.get("website")},
                  context=req.parent_post.text)
    if v.action == "block":   return reject(v.reasons)
    if v.action == "review":  queue_for_moderation(req, v)
    log_verdict(req, v)       # keep for threshold tuning
```

## 2. Signals you should feed it

The behavioral layer is only as good as the metadata you pass. Cheapest wins, in order:

1. **Honeypot field**: a form input hidden with CSS and named something tempting (`website`, `url`). Humans
   never fill it. Pass its value as `meta["honeypot"]`.
2. **`fill_time_s`**: seconds between rendering the form and submitting it (set a signed timestamp in the form).
3. **`user_agent`** from the request. Pass `""` when it is absent so the missing-UA signal can fire.
4. **`account_age_s`** to catch brand-new accounts that post repeatedly.
5. **`context`**: the post/thread being replied to. It unlocks the strongest form of the generic-praise check.
6. **A stable `actor`**: account id when logged in, otherwise IP or device fingerprint. Don't use a
   per-request value or the cadence layer sees one event per actor.

## 3. Thresholds and cost

```python
from aiflytrap.detector import Config
det = SpamDetector.with_default_model(config=Config(review_at=0.35, block_at=0.8))
```

Pick `block_at` from the cost of blocking a real user versus letting spam through, using the precision tables
in [EVALUATION.md](EVALUATION.md) §5 as a guide to how base rate changes things. A sound rollout:

1. **Shadow mode**: log verdicts, take no action, for a week or two.
2. Label a sample from each band (especially `review` and just-below-`review`).
3. Choose thresholds that meet your target false-positive rate on *your* labels.
4. Enable `review` first; enable `block` once the review queue shows few false positives.

Other `Config` fields: `model_blend` (weight of learned vs rule text score), `behavior_weight`,
`duplicate_weight` (set `0` to disable the layer), `dup_floor`, `cross_weight`, `judge_weight`.

## 4. Train the model on your data

The bundled model is trained on the synthetic benchmark. Replace it with one trained on your labeled traffic:

```python
from aiflytrap.features import vectorize
from aiflytrap.model import TextModel

X = [vectorize(r["text"], r.get("context")) for r in rows]     # rows: your labeled messages
y = [r["is_spam"] for r in rows]                               # 1 = text itself is spam
model = TextModel.fit(X, y, epochs=400, l2=1e-2)
model.save("my_model.json")

det = SpamDetector(model="my_model.json")
```

Notes:
- Train the text model only on samples whose **text** is distinguishable. Label stealth-agent messages that
  look normal as `0`, or leave them out. Behavior and duplication layers cover them.
- Include plenty of *hard negatives*: your real power users, AI-assisted benign replies, messages with
  legitimate links. The most common failure of a text model is learning a shortcut like "has a link".
- Weights are plain JSON; open `my_model.json` and check the largest weights for sanity before shipping.
- If you add or change features in `features.py`, retrain. Loading a model with a different feature list
  raises `ValueError` on purpose.

## 5. Add an LLM judge for borderline cases

```python
def judge(text, verdict):
    # call your LLM; return P(spam) in [0, 1], or None to abstain
    ...

det = SpamDetector.with_default_model(judge=judge)
```

The judge is only called when `review_at <= risk < block_at`, so cost scales with the uncertain fraction of
traffic, not total volume. Cache by text hash, set a timeout, and return `None` on any error (the detector
then keeps its own verdict). Treat the message text as untrusted input to your prompt: it may contain
instructions aimed at the judge.

## 6. Operations

- **Memory**: per actor, up to 200 timestamps for 1 hour; the duplicate index holds 5,000 fingerprints.
  Both are bounded.
- **Restart behavior**: state is in memory. After a restart, cadence and duplicate evidence rebuild from
  scratch. For persistence, replay recent events through `score()` at startup.
- **Concurrency**: `score()` is thread-safe (one lock around shared state). `python -m aiflytrap serve` uses a
  threading HTTP server, bound to `127.0.0.1` only; put your own authenticating gateway in front of it.
- **Monitoring**: track the share of `review`/`block` over time. A sudden jump is either an attack or a
  regression. Track the share of `block` verdicts that humans later overturn.
- **Retraining cadence**: attackers change phrasing. Refresh the model and the phrase lexicon in
  `text.py` (`LLM_PHRASES`, `PROMO_WORDS`) from newly flagged messages quarterly or after incidents.
- **Privacy**: scoring is local; no text leaves the process unless you add a judge. Store verdicts and
  reasons for audits, and apply your retention policy to the message text.

## 7. Extending

| To add | Where |
|---|---|
| a new rule signal | function in `text.py` returning `(score, reason)`; register it in `AI_SIGNALS` or `analyze()` |
| a new model feature | key in `extract()` and `FEATURE_NAMES` (`features.py`), then retrain |
| link reputation feed | replace `url_risk()` in `text.py` |
| new behavioral signal | add to `BehaviorTracker.analyze()` and its `meta` keys |
| a stronger duplicate index | implement the `check_and_add(sender, text)` contract of `DuplicateIndex` with LSH |
| new benchmark scenario | add pools and a generator in `synth.py`, list it in `evaluate.CATS_ORDER`, rerun `bench` |

After any change run `python3 -m unittest discover -s tests`: the benchmark gates will tell you if you made
the detector worse on a known scenario.
