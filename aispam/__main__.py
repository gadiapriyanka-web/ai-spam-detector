"""CLI.

  python -m aispam demo
  python -m aispam score "some text" [--model]
  python -m aispam batch events.jsonl [--model]   # {"text","actor","ts","meta","context"} per line
  python -m aispam serve --port 8080 [--model]    # POST /score with the same JSON
  python -m aispam train                          # retrain bundled model on the synthetic train split
  python -m aispam bench                          # run the benchmark, write docs/RESULTS.md
  python -m aispam gen --split test > data.jsonl  # export the synthetic benchmark
"""
import argparse
import json
import os
import sys

from .detector import DEFAULT_MODEL_PATH, SpamDetector

DEMO = [
    ("alice", 0, "Ran into the same bug last week - turned out the cache key included a timestamp. Fixed it in 3 lines, PR #412.", {}),
    ("bot1", 100, "I hope this message finds you well! I came across your insightful profile and was impressed by your work. "
                  "Our cutting-edge platform can unlock the full potential of your business. Furthermore, it is a game-changer. "
                  "Feel free to reach out to book a call: https://example-growth.biz/offer", {"fill_time_s": 0.8, "user_agent": "python-requests/2.31"}),
    ("bot2", 130, "Hi Sam, we help agencies like Acme grow revenue 3x with our growth platform. Book a demo at https://a-growth.biz/go today.", {"fill_time_s": 9}),
    ("bot3", 160, "Hi Priya, we enable agencies such as Globex to scale revenue 5x using our growth platform. Schedule a demo today: https://b-growth.biz/go", {"fill_time_s": 11}),
    ("carol", 200, "Great post! Very informative, thanks for sharing.", {"fill_time_s": 14}),
]


def _detector(args):
    if getattr(args, "model", False):
        if not os.path.exists(DEFAULT_MODEL_PATH):
            sys.exit("no bundled model; run `python -m aispam train` first")
        return SpamDetector.with_default_model()
    return SpamDetector()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aispam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--model", action="store_true")
    s = sub.add_parser("score"); s.add_argument("text"); s.add_argument("--model", action="store_true")
    b = sub.add_parser("batch"); b.add_argument("file"); b.add_argument("--model", action="store_true")
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8080); v.add_argument("--model", action="store_true")
    sub.add_parser("train")
    sub.add_parser("bench")
    g = sub.add_parser("gen"); g.add_argument("--split", default="test"); g.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        from .evaluate import train_model
        os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
        m = train_model()
        m.save(DEFAULT_MODEL_PATH)
        print("saved", DEFAULT_MODEL_PATH)
        for n, w in sorted(zip(m.names, m.w), key=lambda t: -abs(t[1]))[:8]:
            print(f"  {n:24} {w:+.2f}")
        return
    if a.cmd == "bench":
        from .evaluate import main as bench, to_markdown
        os.makedirs("docs", exist_ok=True)
        r = bench(model_path=DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else None)
        print(to_markdown(r))
        return
    if a.cmd == "gen":
        from .synth import gen_stream
        for e in gen_stream(a.split, a.seed):
            print(json.dumps(e))
        return

    det = _detector(a)
    if a.cmd == "demo":
        for actor, ts, text, meta in DEMO:
            r = det.score(text, actor, ts, meta)
            print(f"[{r.action.upper():6}] risk={r.risk:.2f} ai={r.ai_likelihood:.2f} spam={r.spam_likelihood:.2f} "
                  f"auto={r.automation_likelihood:.2f} dup={r.duplication:.2f}  {actor}: {text[:60]}...")
            for why in r.reasons:
                print("          -", why)
    elif a.cmd == "score":
        print(json.dumps(det.score(a.text).to_dict(), indent=2))
    elif a.cmd == "batch":
        with open(a.file) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    r = det.score(e["text"], e.get("actor", "anon"), e.get("ts"), e.get("meta"), e.get("context"))
                    print(json.dumps({"id": e.get("id"), **r.to_dict()}))
    elif a.cmd == "serve":
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/score":
                    self.send_error(404); return
                try:
                    e = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                    out = det.score(e["text"], e.get("actor", "anon"), e.get("ts"), e.get("meta"), e.get("context")).to_dict()
                except (ValueError, KeyError) as ex:
                    self.send_error(400, str(ex)); return
                body = json.dumps(out).encode()
                self.send_response(200); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        print(f"listening on 127.0.0.1:{a.port}", file=sys.stderr)
        ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
