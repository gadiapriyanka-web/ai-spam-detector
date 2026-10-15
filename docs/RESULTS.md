# Benchmark results (auto-generated)

Held-out test stream: **2241 events**, 832 spam (37.1%). Thresholds: review ≥ 0.4, block ≥ 0.7. Final system: **Full (rules + model + behavior + dup)**.

## Ablation: what each layer contributes

| System | ROC-AUC | Avg precision | Precision | Recall | F1 | FPR | Recall @ 1% FPR |
|---|---|---|---|---|---|---|---|
| Text rules only | 0.789 | 0.812 | 100.0% | 23.8% | 0.384 | 0.0% | 61.2% |
| Behavior only | 0.814 | 0.827 | 100.0% | 45.7% | 0.627 | 0.0% | 61.4% |
| Duplication only | 0.589 | 0.435 | 38.6% | 10.2% | 0.162 | 9.6% | 0.0% |
| Rules fusion (no model) | 0.977 | 0.968 | 98.6% | 66.6% | 0.795 | 0.6% | 73.9% |
| Learned text model only | 0.797 | 0.817 | 100.0% | 59.9% | 0.749 | 0.0% | 63.9% |
| Full (rules + model + behavior + dup) | 0.997 | 0.996 | 99.3% | 88.7% | 0.937 | 0.4% | 93.5% |
| Full minus duplication | 1.000 | 0.999 | 100.0% | 83.4% | 0.910 | 0.0% | 99.5% |
| Full minus behavior | 0.965 | 0.965 | 99.0% | 60.2% | 0.749 | 0.4% | 69.0% |

_Precision/recall/F1/FPR are at the block threshold. Single-layer rows use that layer's own score at the same threshold._

## Per-category outcomes (final system)

| Category | Type | Events | Blocked | Flagged (review+block) | Mean risk |
|---|---|---|---|---|---|
| human_casual | legit | 594 | 0.0% | 0.0% | 0.10 |
| human_formal | legit | 222 | 0.0% | 0.0% | 0.03 |
| ai_assisted_benign | legit | 181 | 2.8% | 16.6% | 0.19 |
| human_praise | legit | 100 | 0.0% | 0.0% | 0.04 |
| human_chat_burst | legit | 136 | 0.0% | 2.9% | 0.05 |
| human_link | legit | 176 | 0.0% | 1.7% | 0.21 |
| classic_spam | spam | 120 | 85.8% | 99.2% | 0.82 |
| ai_outreach | spam | 143 | 100.0% | 100.0% | 0.96 |
| ai_comment | spam | 60 | 100.0% | 100.0% | 0.94 |
| ai_seo | spam | 100 | 100.0% | 100.0% | 0.96 |
| paraphrase_campaign | spam | 120 | 72.5% | 100.0% | 0.73 |
| agent_stealth | spam | 289 | 84.8% | 96.2% | 0.83 |

## Time to detection (spam senders)

| Category | Senders | Blocked on first message | Median messages sent before first block | Never blocked |
|---|---|---|---|---|
| ai_outreach | 30 | 100.0% | 0.0 | 0 |
| paraphrase_campaign | 120 | 72.5% | 0 | 33 |
| ai_seo | 25 | 100.0% | 0 | 0 |
| classic_spam | 120 | 85.8% | 0 | 17 |
| agent_stealth | 30 | 80.0% | 0.0 | 0 |
| ai_comment | 30 | 100.0% | 0.0 | 0 |

## Threshold sweep (final system)

| Threshold | Precision | Recall | FPR | Precision if 5% of traffic is spam | Precision if 1% is spam |
|---|---|---|---|---|---|
| 0.2 | 72.6% | 99.9% | 22.3% | 19.1% | 4.3% |
| 0.3 | 84.5% | 99.4% | 10.8% | 32.7% | 8.5% |
| 0.4 | 95.7% | 98.6% | 2.6% | 66.4% | 27.5% |
| 0.5 | 97.8% | 94.6% | 1.3% | 79.6% | 42.8% |
| 0.6 | 98.8% | 92.9% | 0.6% | 88.4% | 59.5% |
| 0.7 | 99.3% | 88.7% | 0.4% | 92.9% | 71.6% |
| 0.8 | 100.0% | 76.0% | 0.0% | 100.0% | 100.0% |
| 0.9 | 100.0% | 50.7% | 0.0% | 100.0% | 100.0% |

## Stability across 5 re-drawn test streams (seeds [7, 11, 13, 17, 19])

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ROC-AUC | 0.998 | 0.001 | 0.997 | 0.999 |
| Avg precision | 0.996 | 0.001 | 0.995 | 0.998 |
| Precision | 99.0% | 0.4% | 98.5% | 99.5% |
| Recall | 87.4% | 2.0% | 85.0% | 90.5% |
| F1 | 0.928 | 0.013 | 0.913 | 0.948 |
| FPR | 0.5% | 0.2% | 0.3% | 0.8% |
| Recall @ 1% FPR | 92.3% | 2.0% | 90.6% | 95.7% |

## Runtime (single thread, pure Python)

2241 messages: mean 1.018 ms, p50 1.041 ms, p95 1.803 ms, p99 2.257 ms, about **982 messages/s**.
