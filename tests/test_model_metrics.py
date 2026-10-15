import math
import os
import random
import tempfile
import unittest

from aispam import metrics as M
from aispam.features import FEATURE_NAMES, extract, vectorize
from aispam.model import TextModel


class MetricsTests(unittest.TestCase):
    def test_auc_known_value(self):
        self.assertAlmostEqual(M.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)

    def test_auc_perfect_and_inverted(self):
        self.assertEqual(M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_auc_ties_use_average_rank(self):
        self.assertAlmostEqual(M.roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_auc_single_class_is_nan(self):
        self.assertTrue(math.isnan(M.roc_auc([1, 1], [0.1, 0.2])))

    def test_average_precision_known_value(self):
        self.assertAlmostEqual(M.average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), (1 + 2 / 3) / 2)

    def test_prf_counts(self):
        m = M.prf([1, 1, 0, 0, 1], [0.9, 0.2, 0.8, 0.1, 0.7], 0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["fpr"], 0.5)

    def test_precision_at_prevalence(self):
        # 90% recall, 1% FPR, 1% spam -> precision ~ 47.6%
        self.assertAlmostEqual(M.precision_at_prevalence(0.9, 0.01, 0.01), 0.009 / (0.009 + 0.0099), places=4)
        self.assertEqual(M.precision_at_prevalence(0.0, 0.0, 0.01), 0.0)

    def test_recall_at_fpr(self):
        y = [1, 1, 1, 0, 0, 0, 0, 0]
        s = [0.9, 0.8, 0.3, 0.4, 0.2, 0.1, 0.1, 0.05]
        self.assertAlmostEqual(M.recall_at_fpr(y, s, 0.0), 2 / 3)


class FeatureAndModelTests(unittest.TestCase):
    def test_vector_matches_names(self):
        v = vectorize("Hello there, this is a short test message! https://x.io")
        self.assertEqual(len(v), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(x) for x in v))

    def test_features_on_empty_text_are_finite(self):
        self.assertTrue(all(math.isfinite(x) for x in vectorize("")))

    def test_features_capture_basic_properties(self):
        f = extract("WIN BIG!!! Claim now https://x.biz https://y.biz")
        self.assertEqual(f["url_count"], 2.0)
        self.assertGreater(f["exclaim_rate"], 0.2)
        self.assertGreater(f["allcaps_rate"], 0.1)
        self.assertEqual(extract("a", "ctx")["has_context"], 1.0)

    def test_fit_learns_separable_problem(self):
        rng = random.Random(1)
        X, y = [], []
        for _ in range(200):
            lab = rng.random() < 0.5
            v = [rng.gauss(0, 1) for _ in FEATURE_NAMES]
            v[FEATURE_NAMES.index("url_count")] += 3 if lab else 0
            X.append(v); y.append(int(lab))
        m = TextModel.fit(X, y, epochs=150)
        acc = sum((m.predict_vec(x) > 0.5) == bool(t) for x, t in zip(X, y)) / len(y)
        self.assertGreater(acc, 0.9)
        top = max(zip(m.names, m.w), key=lambda t: abs(t[1]))[0]
        self.assertEqual(top, "url_count")

    def test_save_load_roundtrip(self):
        m = TextModel.fit([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)], [0, 1], epochs=20)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m.save(p)
            m2 = TextModel.load(p)
        self.assertAlmostEqual(m.predict("click here https://x.biz now"), m2.predict("click here https://x.biz now"))

    def test_load_rejects_feature_mismatch(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump({"names": ["x"], "weights": [0], "bias": 0, "mean": [0], "std": [1]}, f)
            with self.assertRaises(ValueError):
                TextModel.load(p)

    def test_explain_returns_top_features(self):
        m = TextModel.fit([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)], [0, 1], epochs=20)
        self.assertEqual(len(m.explain("hello", None, 3)), 3)


if __name__ == "__main__":
    unittest.main()


class CurveAndBootstrapTests(unittest.TestCase):
    def test_roc_curve_endpoints_and_area(self):
        y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
        pts = M.roc_curve(y, s)
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertEqual(pts[-1], (1.0, 1.0))
        area = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
        self.assertAlmostEqual(area, M.roc_auc(y, s))

    def test_pr_curve_matches_precision_recall(self):
        pts = M.pr_curve([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
        self.assertEqual(pts[0], (0.5, 1.0))
        self.assertEqual(pts[-1], (1.0, 0.5))

    def test_thin_keeps_endpoints_and_shrinks(self):
        pts = [(i / 1000, i / 1000) for i in range(1001)]
        t = M.thin(pts)
        self.assertEqual((t[0], t[-1]), (pts[0], pts[-1]))
        self.assertLess(len(t), 400)

    def test_cluster_bootstrap_brackets_point_estimate(self):
        rng = random.Random(0)
        y, s, g = [], [], []
        for k in range(60):
            lab = k % 2
            for _ in range(5):
                y.append(lab); s.append(rng.random() * 0.6 + 0.4 * lab); g.append(k)
        r = M.cluster_bootstrap(y, s, g, {"auc": M.roc_auc}, n_boot=200, seed=1)["auc"]
        self.assertLessEqual(r[1], r[0])
        self.assertGreaterEqual(r[2], r[0] - 0.02)
        self.assertGreater(r[2], r[1])
