import unittest

from aispam import text as T


class TextSignalTests(unittest.TestCase):
    def test_llm_phrases_fire_on_cliches(self):
        s, why = T.llm_phrases("I hope this message finds you well! It's important to note that we leverage seamless, cutting-edge tools.")
        self.assertGreater(s, 0.6)
        self.assertIn("LLM-typical", why)

    def test_llm_phrases_quiet_on_plain_text(self):
        self.assertEqual(T.llm_phrases("Fixed the cache key bug in three lines, PR 412 is up.")[0], 0.0)

    def test_burstiness_flags_uniform_sentences(self):
        uniform = "This tool helps teams work faster. It also keeps projects well organized. Every user gets simple reports. Teams enjoy clear weekly updates."
        s, _ = T.burstiness(uniform)
        self.assertGreater(s, 0.4)

    def test_burstiness_ignores_varied_and_short_text(self):
        varied = "No. I tried it twice on the old laptop and it still crashed before the login screen even appeared. Weird. Reboot fixed it, somehow, until Friday when the whole thing came back."
        self.assertLess(T.burstiness(varied)[0], 0.3)
        self.assertEqual(T.burstiness("Too short.")[0], 0.0)

    def test_low_specificity_needs_length(self):
        self.assertEqual(T.low_specificity("vague words only")[0], 0.0)
        vague = " ".join(["this solution provides great value and helps people achieve better outcomes every day"] * 3)
        self.assertGreater(T.low_specificity(vague)[0], 0.4)

    def test_specific_text_not_flagged_as_vague(self):
        specific = ("We measured 40ms p99 on a 12 node Postgres cluster running Ubuntu 22.04, and the Grafana dashboard showed "
                    "3 spikes after Tuesday's deploy of version 2.4.1 from Maria's team.")
        self.assertLess(T.low_specificity(specific)[0], 0.3)

    def test_evasion_zero_width_and_homoglyph(self):
        self.assertEqual(T.evasion_chars("bu​y now")[0], 1.0)
        self.assertEqual(T.evasion_chars("pаypal login")[0], 1.0)  # Cyrillic 'a' inside a Latin word
        self.assertEqual(T.evasion_chars("plain ascii text")[0], 0.0)

    def test_promo_counts_links_and_phrases(self):
        s, why = T.promo("Limited time! Click here https://x.biz/offer for a free trial")
        self.assertGreater(s, 0.8)
        self.assertEqual(T.promo("See you at the standup tomorrow")[0], 0.0)

    def test_generic_praise_uses_context_overlap(self):
        ctx = "A deep dive into database indexing: how we cut latency by 40% using Postgres."
        on_topic = "The database indexing section explained why our Postgres latency kept spiking, thanks for the detail."
        off_topic = "Absolutely wonderful perspective and such a beautifully written piece, keep inspiring everyone around you."
        self.assertLess(T.generic_praise(on_topic, ctx)[0], 0.3)
        self.assertGreater(T.generic_praise(off_topic, ctx)[0], 0.5)

    def test_analyze_short_text_dampens_ai_score(self):
        r = T.analyze("Furthermore, seamless!")
        self.assertLess(r["ai"], 0.6)

    def test_analyze_returns_all_signals(self):
        r = T.analyze("hello there friend, nice day")
        for k in ("llm_phrases", "burstiness", "punctuation_tics", "low_specificity", "evasion_chars", "promo", "generic_praise"):
            self.assertIn(k, r["signals"])
        self.assertTrue(0 <= r["ai"] <= 1 and 0 <= r["spam"] <= 1)


if __name__ == "__main__":
    unittest.main()
