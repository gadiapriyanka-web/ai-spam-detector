"""CLI.

  python -m aispam demo
  python -m aispam score "some text"
  python -m aispam batch events.jsonl      # {"text","actor","ts","meta","context"} per line
  python -m aispam serve --port 8080       # POST /score with the same JSON
"""
import argparse
import json
import sys

from .detector import SpamDetector

DEMO = [
    ("alice", 0, "Ran into the same bug last week - turned out the cache key included a timestamp. Fixed it in 3 lines, PR #412.", {}),
    ("bot1", 100, "I hope this message finds you well! I came across your insightful profile and was impressed by your work. "
                  "Our cutting-edge platform can unlock the full potential of your business. Furthermore, it is a game-changer. "
                  "Feel free to reach out to book a call: https://example-growth.biz/offer", {"fill_time_s": 0.8, "user_agent": "python-requests/2.31"}),
    ("bot2", 130, "I hope this message finds you well! I came across your insightful company and was impressed by your work. "
                  "Our cutting-edge tool can unlock the full potential of your brand. Moreover, it is a game changer. "
                  "Don't hesitate to book a call: https://another-growth.biz/deal", {"fill_time_s": 1.1}),
    ("bot3", 160, "Great post! Very informative, thanks for sharing. Keep up the great work!", {}),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aispam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    s = sub.add_parser("score"); s.add_argument("text")
    b = sub.add_parser("batch"); b.add_argument("file")
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    det = SpamDetector()

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
        from http.server import BaseHTTPRequestHandler, HTTPServer

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

        print(f"listening on :{a.port}", file=sys.stderr)
        HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
