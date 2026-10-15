import unittest

from aispam import SpamDetector

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


if __name__ == "__main__":
    unittest.main()
