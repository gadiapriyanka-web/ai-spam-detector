import unittest

from aiflytrap.behavior import BehaviorTracker


class BehaviorTests(unittest.TestCase):
    def test_honeypot_is_certain(self):
        r = BehaviorTracker().analyze("a", 0, {"honeypot": "http://x"})
        self.assertEqual(r["score"], 1.0)

    def test_fast_form_fill(self):
        fast = BehaviorTracker().analyze("a", 0, {"fill_time_s": 0.4})["score"]
        slow = BehaviorTracker().analyze("b", 0, {"fill_time_s": 45})["score"]
        self.assertGreater(fast, 0.5)
        self.assertEqual(slow, 0.0)

    def test_automation_user_agents(self):
        for ua in ("python-requests/2.31", "HeadlessChrome/120", "Go-http-client/1.1", "Playwright", "langchain-agent/0.1"):
            self.assertGreater(BehaviorTracker().analyze("a", 0, {"user_agent": ua})["score"], 0.5, ua)
        browser = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15"
        self.assertEqual(BehaviorTracker().analyze("a", 0, {"user_agent": browser})["score"], 0.0)

    def test_missing_user_agent_is_mildly_suspicious(self):
        s = BehaviorTracker().analyze("a", 0, {"user_agent": ""})["score"]
        self.assertTrue(0.2 < s < 0.7)

    def test_metronomic_cadence(self):
        t = BehaviorTracker()
        r = None
        for i in range(8):
            r = t.analyze("bot", i * 60.0)
        self.assertGreater(r["signals"]["regularity"], 0.9)
        self.assertGreater(r["score"], 0.5)

    def test_bursty_human_cadence(self):
        t = BehaviorTracker()
        r = None
        for ts in (0, 12, 15, 900, 2400, 2420):
            r = t.analyze("human", ts)
        self.assertLess(r["signals"].get("regularity", 0), 0.3)

    def test_high_rate(self):
        t = BehaviorTracker()
        r = None
        for i in range(30):
            r = t.analyze("flood", i * 1.0 + (i % 3) * 0.4)
        self.assertGreater(r["signals"]["rate"], 0.8)

    def test_actors_are_isolated(self):
        t = BehaviorTracker()
        for i in range(8):
            t.analyze("bot", i * 60.0)
        self.assertEqual(t.analyze("someone-else", 500)["score"], 0.0)

    def test_window_expiry(self):
        t = BehaviorTracker(window_seconds=100)
        for i in range(6):
            t.analyze("a", i * 10.0)
        t.analyze("a", 10_000)
        self.assertEqual(len(t.events["a"]), 1)

    def test_new_account_burst(self):
        t = BehaviorTracker()
        r = None
        for ts in (0, 20, 95):
            r = t.analyze("n", ts, {"account_age_s": 120})
        self.assertIn("new_account_burst", r["signals"])

    def test_score_bounded(self):
        t = BehaviorTracker()
        r = None
        for i in range(10):
            r = t.analyze("x", i * 1.0, {"fill_time_s": 0.1, "user_agent": "curl/8", "account_age_s": 5})
        self.assertTrue(0 <= r["score"] <= 1)


if __name__ == "__main__":
    unittest.main()
