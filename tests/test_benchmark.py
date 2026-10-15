"""Regression gates on the synthetic benchmark.

Thresholds sit a little below current performance so they act as tripwires: if a signal or weight change makes these fail,
it made the detector worse on at least one known scenario. They are not claims about real-world accuracy.
"""
import unittest
from collections import Counter

from aispam import metrics as M
from aispam.detector import DEFAULT_MODEL_PATH, SpamDetector
from aispam.evaluate import run_stream
from aispam.synth import LEGIT, SPAM, gen_stream


class SynthTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(gen_stream("test", 3, 0.2), gen_stream("test", 3, 0.2))

    def test_seeds_differ(self):
        self.assertNotEqual([e["text"] for e in gen_stream("test", 3, 0.2)], [e["text"] for e in gen_stream("test", 4, 0.2)])

    def test_chronological_and_labelled(self):
        ev = gen_stream("test", 1, 0.2)
        self.assertEqual([e["ts"] for e in ev], sorted(e["ts"] for e in ev))
        for e in ev:
            self.assertEqual(e["label"], int(e["category"] in SPAM))
        self.assertTrue({e["category"] for e in ev} <= (SPAM | LEGIT))

    def test_train_and_test_wording_do_not_overlap_for_fixed_templates(self):
        # classic spam templates are fixed strings with slots: no template text may appear in both splits
        strip = lambda t: "".join(c for c in t if c.isalpha())[:25]
        tr = {strip(e["text"]) for e in gen_stream("train", 1, 0.5) if e["category"] == "classic_spam"}
        te = {strip(e["text"]) for e in gen_stream("test", 1, 0.5) if e["category"] == "classic_spam"}
        self.assertFalse(tr & te)


class BenchmarkGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = gen_stream("test", seed=23, scale=0.5)
        v, _ = run_stream(SpamDetector.with_default_model(), cls.ev)
        cls.risk = [x.risk for x in v]
        cls.y = [e["label"] for e in cls.ev]

    def test_ranking_quality(self):
        self.assertGreater(M.roc_auc(self.y, self.risk), 0.97)

    def test_operating_point(self):
        m = M.prf(self.y, self.risk, 0.7)
        self.assertGreater(m["precision"], 0.95)
        self.assertGreater(m["recall"], 0.80)
        self.assertLess(m["fpr"], 0.03)

    def test_no_legit_category_is_mostly_blocked(self):
        by = Counter(); n = Counter()
        for e, r in zip(self.ev, self.risk):
            if e["category"] in LEGIT:
                n[e["category"]] += 1
                by[e["category"]] += r >= 0.7
        for c in n:
            self.assertLess(by[c] / n[c], 0.06, c)

    def test_every_spam_family_mostly_caught(self):
        blocked = Counter(); flagged = Counter(); n = Counter()
        for e, r in zip(self.ev, self.risk):
            if e["category"] in SPAM:
                n[e["category"]] += 1
                blocked[e["category"]] += r >= 0.7
                flagged[e["category"]] += r >= 0.4
        for c in n:
            self.assertGreater(blocked[c] / n[c], 0.55, f"{c}: block rate")
            self.assertGreater(flagged[c] / n[c], 0.90, f"{c}: flag rate")

    def test_rules_only_still_works_without_model(self):
        v, _ = run_stream(SpamDetector(model=None), self.ev)
        self.assertGreater(M.roc_auc(self.y, [x.risk for x in v]), 0.93)

    def test_speed_budget(self):
        import time
        d = SpamDetector.with_default_model()
        t0 = time.perf_counter()
        for i, e in enumerate(self.ev[:300]):
            d.score(e["text"], e["actor"], e["ts"], e["meta"], e["context"])
        per = (time.perf_counter() - t0) / 300
        self.assertLess(per, 0.01)  # < 10 ms per message, ~10x headroom over measured


if __name__ == "__main__":
    unittest.main()
