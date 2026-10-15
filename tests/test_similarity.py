import unittest

from aispam.similarity import DuplicateIndex, hamming, simhash, skeleton

T1 = "Hi Sam, we help agencies like Acme grow revenue 3x with our growth platform. Book a demo at https://a.biz/go today."
T2 = "Hi Priya, we help agencies like Globex grow revenue 5x with our growth platform. Book a demo at https://b.biz/go today."
UNRELATED = "Honestly the build kept failing on my laptop until I cleared the cache directory and reinstalled the toolchain from scratch."


class SimilarityTests(unittest.TestCase):
    def test_simhash_close_for_near_duplicates(self):
        self.assertLess(hamming(simhash(T1), simhash(T2)), hamming(simhash(T1), simhash(UNRELATED)))

    def test_skeleton_ignores_names_numbers_links(self):
        self.assertEqual(skeleton(T1), skeleton(T2))
        self.assertNotEqual(skeleton(T1), skeleton(UNRELATED))

    def test_campaign_across_senders_scores_higher_with_more_senders(self):
        d = DuplicateIndex()
        d.check_and_add("a", T1)
        s1 = d.check_and_add("b", T2)["score"]
        s2 = d.check_and_add("c", T1.replace("Sam", "Lee"))["score"]
        s3 = d.check_and_add("e", T2.replace("Priya", "Ana"))["score"]
        self.assertTrue(s1 < s2 <= s3)
        self.assertGreater(s3, 0.6)

    def test_unrelated_text_scores_zero(self):
        d = DuplicateIndex()
        d.check_and_add("a", T1)
        self.assertEqual(d.check_and_add("b", UNRELATED)["score"], 0.0)

    def test_short_messages_are_ignored(self):
        d = DuplicateIndex()
        for who in "abcde":
            r = d.check_and_add(who, "thanks for sharing!")
        self.assertEqual(r["score"], 0.0)

    def test_exact_self_repeat_not_counted_here(self):
        d = DuplicateIndex()
        d.check_and_add("a", T1)
        self.assertEqual(d.check_and_add("a", T1)["matches"], 0)

    def test_capacity_bounds_memory(self):
        d = DuplicateIndex(capacity=10)
        for i in range(50):
            d.check_and_add(f"u{i}", f"message number {i} with enough words to count as a real message here")
        self.assertEqual(len(d.items), 10)


if __name__ == "__main__":
    unittest.main()
