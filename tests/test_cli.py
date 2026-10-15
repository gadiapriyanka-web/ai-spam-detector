import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from aispam.__main__ import main


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(list(argv))
        return buf.getvalue()

    def test_demo(self):
        out = self.run_cli("demo")
        self.assertIn("[ALLOW ]", out)
        self.assertIn("[BLOCK ]", out)

    def test_score_json(self):
        d = json.loads(self.run_cli("score", "Click here https://x.biz for a free trial, limited time!"))
        self.assertIn(d["action"], {"allow", "review", "block"})

    def test_batch_roundtrip(self):
        rows = [{"id": 1, "text": "Ran into the same bug last week, fixed it in 3 lines.", "actor": "a", "ts": 0},
                {"id": 2, "text": "Buy now!!! limited time https://x.biz click here", "actor": "b", "ts": 1}]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "in.jsonl")
            with open(p, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            out = [json.loads(l) for l in self.run_cli("batch", p).splitlines()]
        self.assertEqual([o["id"] for o in out], [1, 2])
        self.assertEqual(out[0]["action"], "allow")

    def test_gen_emits_jsonl(self):
        first = json.loads(self.run_cli("gen", "--split", "test").splitlines()[0])
        self.assertIn("label", first)


if __name__ == "__main__":
    unittest.main()
