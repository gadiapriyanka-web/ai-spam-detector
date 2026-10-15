import threading
import unittest

from aispam import SpamDetector
from aispam.detector import Config, DEFAULT_MODEL_PATH

HUMAN = "Ran into the same bug last week - turned out the cache key included a timestamp. Fixed it in 3 lines, PR #412."
AI_SPAM = ("I hope this message finds you well! I came across your insightful profile and was impressed by your work. "
           "Our cutting-edge platform can unlock the full potential of your business. Furthermore, it is a game-changer. "
           "Feel free to reach out to book a call: https://example-growth.biz/offer")


class DetectorTests(unittest.TestCase):
    def test_human_allowed(self):
        self.assertEqual(SpamDetector().score(HUMAN, "h", 0).action, "allow")

    def test_ai_spam_blocked(self):
        r = SpamDetector().score(AI_SPAM, "s", 0, {"fill_time_s": 0.5})
        self.assertEqual(r.action, "block", r)

    def test_ai_text_alone_not_blocked(self):
        polite = ("Thank you for the detailed write-up. Furthermore, the benchmark numbers on page 4 line up with what "
                  "we measured on our staging cluster, which had 12 nodes and a 40ms median latency.")
        self.assertNotEqual(SpamDetector().score(polite, "p", 0).action, "block")

    def test_honeypot(self):
        self.assertEqual(SpamDetector().score(HUMAN, "h", 0, {"honeypot": "x"}).action, "block")

    def test_metronomic_agent(self):
        d = SpamDetector()
        r = None
        for i in range(8):
            r = d.score(f"Interesting point number {i} about your project timeline and budget.", "agent", i * 60.0)
        self.assertGreater(r.automation_likelihood, 0.5)

    def test_human_bursty_timing_ok(self):
        d = SpamDetector()
        r = None
        for t in [0, 40, 41, 900, 1500, 1520]:
            r = d.score("a normal reply " + str(t), "human", t)
        self.assertLess(r.automation_likelihood, 0.3)

    def test_paraphrased_campaign_detected(self):
        d = SpamDetector()
        d.score("Hi Sam, we help companies like Acme grow revenue 3x with our platform, book a demo at https://a.io today", "x1", 0)
        r = d.score("Hi Priya, we help companies like Globex grow revenue 5x with our platform, book a demo at https://b.io today", "x2", 5)
        self.assertGreater(r.duplication, 0.3)

    def test_zero_width_evasion(self):
        r = SpamDetector().score("bu​y ch​eap se​o backlinks now at www.x.biz", "z", 0)
        self.assertGreater(r.spam_likelihood, 0.9)

    # ---- new behaviour
    def test_verdict_fields_and_bounds(self):
        v = SpamDetector().score(AI_SPAM, "s", 0).to_dict()
        for k in ("risk", "action", "ai_likelihood", "spam_likelihood", "automation_likelihood", "duplication", "reasons", "signals"):
            self.assertIn(k, v)
        self.assertTrue(0 <= v["risk"] <= 1)

    def test_thresholds_are_configurable(self):
        strict = SpamDetector(config=Config(review_at=0.05, block_at=0.1)).score(AI_SPAM, "s", 0)
        lax = SpamDetector(config=Config(review_at=1.01, block_at=1.02)).score(AI_SPAM, "s2", 0)
        self.assertEqual(strict.action, "block")
        self.assertNotEqual(lax.action, "block")

    def test_trusted_actor_skips_behavior(self):
        d = SpamDetector(trusted_actors={"ci-bot"})
        r = None
        for i in range(10):
            r = d.score(f"Build {i} finished in 4m12s for commit abc{i}def on the main branch", "ci-bot", i * 60.0,
                        {"user_agent": "python-requests/2.31", "fill_time_s": 0.1})
        self.assertEqual(r.automation_likelihood, 0.0)
        self.assertEqual(r.action, "allow")

    def test_judge_only_consulted_in_review_band(self):
        calls = []

        def judge(text, v):
            calls.append(v.risk)
            return 1.0

        d = SpamDetector(judge=judge)
        d.score(HUMAN, "h", 0)            # clearly fine, no judge
        d.score(AI_SPAM, "s", 0, {"fill_time_s": 0.5, "user_agent": "python-requests/2"})  # clearly spam, no judge
        self.assertEqual(calls, [])

    def test_judge_can_escalate_borderline(self):
        base = SpamDetector(config=Config(review_at=0.2, block_at=0.9))
        text = "Great insights! Feel free to visit https://x.biz for more on this."
        r0 = base.score(text, "a", 0)
        self.assertEqual(r0.action, "review", r0)
        d = SpamDetector(config=Config(review_at=0.2, block_at=0.9, judge_weight=0.9), judge=lambda t, v: 1.0)
        r1 = d.score(text, "a", 0)
        self.assertGreater(r1.risk, r0.risk)
        self.assertEqual(r1.action, "block")

    def test_judge_returning_none_is_ignored(self):
        d = SpamDetector(config=Config(review_at=0.2, block_at=0.9), judge=lambda t, v: None)
        self.assertEqual(d.score("Great insights! Feel free to visit https://x.biz for more on this.", "a", 0).action, "review")

    def test_bundled_model_loads_and_is_used(self):
        d = SpamDetector.with_default_model()
        v = d.score(AI_SPAM, "s", 0)
        self.assertIsNotNone(v.text_model_score)
        self.assertGreater(v.text_model_score, 0.5)
        self.assertLess(d.score(HUMAN, "h", 0).text_model_score, 0.3)

    def test_model_accepts_path(self):
        self.assertIsNotNone(SpamDetector(model=DEFAULT_MODEL_PATH).model)

    def test_thread_safety_smoke(self):
        d = SpamDetector()
        errs = []

        def work(k):
            try:
                for i in range(100):
                    d.score(f"message {i} from worker {k} with enough words to be scored properly", f"w{k}", i * 3.0)
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errs, [])

    def test_empty_and_unicode_inputs_do_not_crash(self):
        d = SpamDetector.with_default_model()
        for txt in ("", " ", "​", "🔥" * 50, "a" * 20000, "你好，世界。这是一条测试消息。"):
            v = d.score(txt, "u", 0)
            self.assertTrue(0 <= v.risk <= 1)


class AdversarialTests(unittest.TestCase):
    """Cheap evasions should not flip an obvious verdict."""

    def _spam(self, text, **meta):
        return SpamDetector().score(text, "adv", 0, dict({"fill_time_s": 0.5, "user_agent": "python-requests/2"}, **meta))

    def test_zero_width_padding(self):
        self.assertEqual(self._spam("​".join(AI_SPAM)).action, "block")

    def test_uppercase_does_not_hide_phrases(self):
        self.assertEqual(self._spam(AI_SPAM.upper()).action, "block")

    def test_homoglyph_substitution_is_flagged(self):
        r = self._spam(AI_SPAM.replace("a", "а", 12))
        self.assertGreater(r.spam_likelihood, 0.9)

    def test_clean_human_text_with_link_is_not_blocked(self):
        v = SpamDetector.with_default_model().score(
            "Here is the PR I mentioned, the fix is in the second commit: https://github.com/acme/widgets/pull/412", "h", 0,
            {"fill_time_s": 30, "user_agent": "Mozilla/5.0 (Macintosh)"})
        self.assertNotEqual(v.action, "block")


if __name__ == "__main__":
    unittest.main()
