"""Synthetic benchmark generator.

Produces a chronologically interleaved event stream with ground-truth labels. Every
phrase pool is split in two halves: the "train" split draws from even indices, the
"test" split from odd indices, so the learned model never sees the test wording.

This is *hand-authored* data meant for regression testing and ablations. It is
not a substitute for evaluating on your own labeled traffic (see docs/EVALUATION.md).
"""
from __future__ import annotations

import random
import re

BASE_TS = 1_760_000_000.0
DAY = 86400.0

# categories whose *text alone* is spammy (used as the text-model training target)
TEXT_SPAM = {"ai_outreach", "ai_comment", "ai_seo", "classic_spam", "paraphrase_campaign"}
# spam categories whose text is deliberately indistinguishable from normal text
TEXT_STEALTH = {"agent_stealth"}
LEGIT = {"human_casual", "human_formal", "ai_assisted_benign", "human_praise", "human_chat_burst", "human_link"}
SPAM = TEXT_SPAM | TEXT_STEALTH

SLOTS = {
    "thing": ["memory leak", "flaky test", "timeout", "deadlock", "build failure", "race condition",
              "segfault", "config drift", "cache stampede", "schema mismatch"],
    "tool": ["Redis", "Postgres", "Terraform", "Kubernetes", "Webpack", "pytest", "Grafana",
             "Kafka", "Docker", "nginx", "Ansible", "Prometheus"],
    "os": ["Ubuntu 22.04", "macOS 14", "Windows 11", "Debian 12", "Fedora 39", "Alpine 3.19"],
    "device": ["Raspberry Pi 4", "ThinkPad", "M2 Air", "old Dell", "Steam Deck", "home server"],
    "feature": ["streaming", "partial indexes", "hot reload", "IPv6", "arm64 builds", "batch writes"],
    "day": ["Monday", "Tuesday", "Thursday", "Friday", "Sunday"],
    "name": ["Maria", "Chen", "Tomasz", "Aisha", "Daniel", "Priya", "Lukas", "Sofia", "Kwame", "Elena"],
    "company": ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Hooli", "Stark Works", "Wayne Logistics",
                "Soylent Foods", "Vandelay Imports", "Tyrell Systems"],
    "biztype": ["agencies", "startups", "retailers", "consultancies", "clinics", "restaurants", "SaaS teams"],
    "topic": ["database indexing", "zero-downtime deploys", "container networking", "feature flags",
              "log aggregation", "load testing", "incident reviews", "API versioning"],
    "metric": ["traffic", "rankings", "leads", "followers", "conversions", "domain authority"],
    "service": ["SEO package", "growth platform", "backlink service", "lead-gen toolkit", "outreach suite"],
    "pill": ["watches", "sneakers", "pills", "followers", "handbags", "vapes"],
    "biz": ["business", "brand", "workflow", "pipeline", "team", "operations"],
}
WORDS_SPAM = ["growth", "boost", "leads", "rank", "seo", "pro", "hub", "scale", "reach", "viral", "prime", "max"]
TLDS = ["biz", "co", "io", "xyz", "online", "net"]
UAS_BROWSER = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]
UAS_BOT = ["python-requests/2.31.0", "HeadlessChrome/120.0", "Go-http-client/1.1", "node-fetch/3.3", "axios/1.6"]


def _pick(pool, split, rng):
    sub = pool[0::2] if split == "train" else pool[1::2]
    return rng.choice(sub)


def _fill(s, rng):
    def rep(m):
        k = m.group(1)
        if k == "n":
            return str(rng.choice([3, 4, 7, 12, 18, 24, 40, 64, 120, 250, 480]))
        if k == "pct":
            return str(rng.randint(8, 74))
        if k == "ver":
            return f"{rng.randint(1, 5)}.{rng.randint(0, 19)}.{rng.randint(0, 9)}"
        if k == "ghurl":
            return (f"https://github.com/{rng.choice(['acme', 'octo', 'kube', 'rust-lang', 'pallets'])}/"
                    f"{rng.choice(['widgets', 'core', 'cli', 'runtime', 'docs'])}/{rng.choice(['pull', 'issues'])}/{rng.randint(10, 9999)}")
        if k == "docurl":
            return rng.choice(["https://docs.python.org/3/library/asyncio.html", "https://stackoverflow.com/questions/%d" % rng.randint(10**6, 10**8),
                               "https://en.wikipedia.org/wiki/Race_condition", "https://arxiv.org/abs/2310.%05d" % rng.randint(1, 99999),
                               "https://kubernetes.io/docs/concepts/", "https://www.postgresql.org/docs/16/indexes.html"])
        if k == "url":
            return f"https://{rng.choice(WORDS_SPAM)}-{rng.choice(WORDS_SPAM)}.{rng.choice(TLDS)}/{rng.choice(['offer', 'deal', 'start', 'go', 'join'])}"
        return rng.choice(SLOTS[k])
    return re.sub(r"\{(\w+)\}", rep, s)


def _compose(parts, split, rng, sep=" "):
    """parts: list of (pool, probability)."""
    out = [_pick(pool, split, rng) for pool, p in parts if rng.random() < p]
    return _fill(sep.join(out), rng)


# ------------------------------------------------------------------ metas
def _human_meta(rng, fill=(6, 150)):
    return {"fill_time_s": rng.uniform(*fill), "user_agent": rng.choice(UAS_BROWSER),
            "account_age_s": rng.uniform(3 * DAY, 2000 * DAY)}


def _bot_meta(rng, ua_bot=0.5, fill=(0.3, 3.0)):
    ua = rng.choice(UAS_BOT) if rng.random() < ua_bot else rng.choice(UAS_BROWSER)
    return {"fill_time_s": rng.uniform(*fill), "user_agent": ua, "account_age_s": rng.uniform(60, 3 * DAY)}


# -------------------------------------------------------------- text pools
HUMAN_CASUAL_OPEN = [
    "Ran into the same {thing} last week", "Tried this on my {device} and it took {n} seconds",
    "Honestly not sure about the {pct}% figure", "We hit this at work, our {tool} config was wrong",
    "lol same, mine crashed twice on {day}", "Does anyone know if {tool} supports {feature} yet?",
    "Thanks, that fixed it for me on {os}", "Disagree with point {n}, we measured the opposite",
    "Hmm, got a {thing} after upgrading {tool} to {ver}", "Can confirm on {os}, same {thing}",
]
HUMAN_CASUAL_BODY = [
    "Turned out the cache key included a timestamp.", "Had to downgrade to {ver} to make it work.",
    "Rebooting twice sorted it, no idea why.", "The docs say one thing but the CLI does another.",
    "Took me about {n} minutes to find.", "Switched to {tool} and the problem went away.",
    "Might just be my setup though.", "Pretty sure it's a {thing} in the loader.",
    "Fixed it in {n} lines, PR is up.", "Nobody on the team could reproduce it, which was fun.",
]
HUMAN_FORMAL_GREET = [
    "Thank you for the detailed report.", "Thanks for taking the time to write this up.",
    "I appreciate the thorough explanation.", "Hello {name}, thanks for flagging this.",
    "Good morning, and thanks for the follow-up.", "Thank you for raising this issue.",
]
HUMAN_FORMAL_FIND = [
    "We reproduced the problem on build {ver} using {n} nodes, and median latency was {n}ms.",
    "On {os} with {tool} {ver} the {thing} appears after roughly {n} requests.",
    "Our staging cluster ({n} nodes, {tool} {ver}) shows the same {thing}, though only under load.",
    "I checked the logs from {day}; the {thing} starts about {n} seconds after the deploy finishes.",
    "The numbers in section {n} differ from ours by about {pct}%, probably because we ran {tool} {ver}.",
    "Roughly {pct}% of the failures we saw came from a single {thing} in the {tool} client.",
]
HUMAN_FORMAL_ASK = [
    "Could you share the {tool} config you used?", "Would it be possible to send the exact commit hash?",
    "Do you have a trace from the failing run?", "Can you confirm whether {feature} was enabled?",
    "It would help to know which {os} image you started from.", "Let me know if a heap dump would be useful.",
]
HUMAN_FORMAL_CLOSE = ["Regards, {name}", "Best, {name}", "Thanks again, {name}", "Cheers, {name}", "Kind regards, {name}", "- {name}"]

AI_BENIGN_OPEN = [
    "Thanks for sharing this.", "This is a really helpful write-up.", "Great breakdown of the trade-offs.",
    "I appreciate you laying this out so clearly.", "Thank you for this thoughtful analysis.", "Nice work putting this together.",
]
AI_BENIGN_BODY = [
    "It's important to note that the benchmark in section {n} used {n} threads, so the results may not generalise to {os}.",
    "Furthermore, in our {tool} deployment the {thing} only showed up above {n} requests per second.",
    "Moreover, the {pct}% improvement you report matches what we saw after moving {n} services to {tool} {ver}.",
    "One caveat worth highlighting: our {tool} cluster on {os} behaved differently once {feature} was turned on.",
    "In our experience at {company}, the {thing} disappeared after we pinned {tool} to {ver}.",
    "It's worth noting that we measured {n}ms p99 with {tool}, which is well below the figure in your table.",
]
AI_BENIGN_CLOSE = [
    "I'd be happy to share our numbers if that's useful.", "Feel free to reach out if you'd like the raw logs.",
    "Happy to compare notes on the {tool} setup.", "I can post the {tool} dashboard export if it helps.",
]

PRAISE_GENERIC_HUMAN = ["Great post, thanks for sharing!", "Nice write-up.", "Very informative, thanks!", "Love this, bookmarked.",
                        "Really helpful, thank you.", "This is great, thanks."]
PRAISE_SPECIFIC_HUMAN = [
    "Great post. The part about {topic} was exactly what I needed for our {tool} migration.",
    "Thanks, the {topic} section finally explained why our {thing} kept coming back.",
    "Really liked the {topic} example; we are trying it on {n} services next week.",
    "Bookmarked for the {topic} bit. Did you hit any {thing} while testing it?",
]
HUMAN_LINK = [
    "Here is the PR I mentioned: {ghurl}", "The repro is in this issue, see {ghurl}", "Docs for that flag are here: {docurl}",
    "Related discussion, might help: {docurl}", "Fixed upstream in {ghurl}, should land in {ver}.", "This thread has the workaround: {docurl}",
    "I wrote up the steps in {ghurl} if it helps", "Same {thing} reported at {docurl}", "Spec reference for the {feature} behaviour: {docurl}",
    "Opened {ghurl} with a failing test for the {thing}.",
]
HUMAN_LINK_TAIL = ["Let me know if it works on {os}.", "Took me {n} minutes to track down.", "Not sure it applies to {tool} {ver} though.", "Might save you some time."]
CHAT_BURST = ["yeah", "ok that works", "wait what", "can you resend the link", "nvm found it", "try {tool} instead",
              "same here on {os}", "ha, classic {thing}", "one sec", "pushing a fix now", "that did it, thanks", "which version of {tool}?"]

AI_OUTREACH_OPEN = [
    "I hope this message finds you well!", "I came across your profile and was truly impressed by your work.",
    "I hope you're having a wonderful week!", "I was impressed by your insightful content on LinkedIn.",
    "I hope this email finds you well.", "Your recent work really resonated with me.",
    "I came across your {biz} and wanted to reach out.", "I hope this note finds you in good spirits!",
]
AI_OUTREACH_BODY = [
    "Our cutting-edge platform can unlock the full potential of your {biz}.",
    "We leverage seamless AI-driven solutions to elevate your {biz} to the next level.",
    "In today's fast-paced digital landscape, a holistic approach to {metric} is a game-changer.",
    "Our solution is not just a tool, but a comprehensive ecosystem designed to transform your {metric}.",
    "We help {biztype} navigate the complexities of {metric} with a robust, scalable framework.",
    "Furthermore, our innovative approach delivers measurable results across your entire {biz}.",
    "Moreover, our streamlined workflow will empower you to unleash new growth in {metric}.",
    "By harnessing a myriad of data-driven strategies, we dive into what makes your {biz} thrive.",
]
AI_OUTREACH_CTA = [
    "Feel free to reach out to book a call: {url}", "Don't hesitate to schedule a quick demo at {url}",
    "I'd be happy to share a free trial, just visit {url}", "Let's connect! Book a call here: {url}",
]
AI_COMMENT_GENERIC = [
    "Great insights! This really resonates with me.", "Thanks for sharing this valuable perspective, truly insightful.",
    "What a fantastic and well-written post, keep up the great work!", "This is such an important topic, and you've explained it beautifully.",
    "Absolutely love this content, it's a testament to your expertise.", "Wow, great post! Very informative and thought-provoking.",
]
AI_COMMENT_PLUG = [
    "I recently explored similar ideas on my blog: {url}", "If you're interested in {metric}, check out my guide at {url}",
    "For anyone looking to boost their {metric}, our {service} can help: {url}", "Feel free to visit {url} for more on this.",
]
AI_SEO_OPEN = [
    "Looking to boost your {metric}?", "Want to increase your {metric} fast?", "Struggling to grow your {metric}?", "Ready to elevate your {metric}?",
]
AI_SEO_BODY = [
    "Our {service} delivers guaranteed results with a seamless, cutting-edge approach.",
    "Our {service} unlocks the full potential of your online presence, 100% guaranteed.",
    "With our {service}, you will see a limited time discount on premium backlinks.",
    "Our {service} is a game-changer for {biztype} who want special offer pricing.",
]
AI_SEO_CTA = ["Visit {url} for a special offer today.", "Click here: {url}", "Act now at {url}", "Book a call and get a free trial: {url}"]
CLASSIC = [
    "BUY CHEAP {pill} NOW!!! 100% guaranteed, click here {url}", "Earn ${n} per day working from home!!! DM me on telegram",
    "FREE crypto airdrop, claim now {url}", "cheap {pill} best price visit {url} limited time",
    "hey u won a prize!!! click {url} to claim", "make money fast with crypto casino bonus {url}",
    "Meet hot singles in your area {url}", "ACT NOW special offer on {pill} {url}",
]
STEALTH = [
    "Interesting approach to {topic}. Have you considered {tool} for the {thing}?",
    "We've seen {topic} work well at scale. What was your experience with {tool}?",
    "Good point about {topic}. Curious how this compares with {tool} in production.",
    "Does this still hold when {feature} is enabled? We saw different results with {tool}.",
    "Thanks for the write-up on {topic}. How did you handle the {thing}?",
    "That matches what we observed with {tool} {ver}. Did you test on {os} as well?",
]
CAMPAIGN_OPEN = ["Hi {name}, we help {biztype} like {company} grow revenue {n}x with our {service}.",
                 "Hello {name}, we enable {biztype} such as {company} to scale revenue {n}x using our {service}.",
                 "Hey {name}, our team assists {biztype} like {company} in growing revenue {n}x via our {service}.",
                 "Hi {name}, we support {biztype} like {company} to multiply revenue {n}x with our {service}.",
                 "Dear {name}, we help {biztype} including {company} reach {n}x revenue with our {service}.",
                 "Hello {name}, we empower {biztype} like {company} to hit {n}x revenue through our {service}."]
CAMPAIGN_CTA = ["Book a demo at {url} today.", "Grab a free demo here: {url}", "Schedule a demo today: {url}", "Reply or visit {url} to book a demo.",
                "Claim your demo at {url} this week.", "See it live at {url}, demos are free."]


# ------------------------------------------------------------- generators
def _humans(cat, n, rng, split, gen, post_range=(1, 3), meta_fn=_human_meta, ctxgen=None):
    ev = []
    for i in range(n):
        actor = f"{cat}-{i}"
        t = BASE_TS + rng.uniform(0, DAY)
        for _ in range(rng.randint(*post_range)):
            ctx = ctxgen(rng) if ctxgen else None
            text = gen(rng, split, ctx)
            ev.append(dict(text=text, actor=actor, ts=t, meta=meta_fn(rng), context=ctx, category=cat))
            t += rng.lognormvariate(6.5, 1.2)  # heavy-tailed human gaps (median ~11 min)
    return ev


def _parent(rng):
    return _fill("A deep dive into {topic}: how we cut {thing} by {pct}% using {tool}.", rng)


def gen_stream(split: str = "test", seed: int = 7, scale: float = 1.0) -> list[dict]:
    """Return events sorted by timestamp with ground-truth `label` (1 = spam to block)."""
    rng = random.Random(f"{seed}-{split}")
    S = lambda n: max(2, int(n * scale))
    ev: list[dict] = []

    ev += _humans("human_casual", S(300), rng, split,
                  lambda r, sp, _c: _compose([(HUMAN_CASUAL_OPEN, 1), (HUMAN_CASUAL_BODY, .6)], sp, r, ". ").rstrip(".") + ".")
    ev += _humans("human_formal", S(150), rng, split, post_range=(1, 2), gen=lambda r, sp, _c: _compose(
        [(HUMAN_FORMAL_GREET, 1), (HUMAN_FORMAL_FIND, 1), (HUMAN_FORMAL_ASK, .8), (HUMAN_FORMAL_CLOSE, .8)], sp, r))
    ev += _humans("ai_assisted_benign", S(120), rng, split, post_range=(1, 2), gen=lambda r, sp, _c: _compose(
        [(AI_BENIGN_OPEN, 1), (AI_BENIGN_BODY, 1), (AI_BENIGN_CLOSE, .7)], sp, r))
    ev += _praise(rng, split, S(100))
    ev += _humans("human_link", S(120), rng, split, post_range=(1, 2),
                  gen=lambda r, sp, _c: _compose([(HUMAN_LINK, 1), (HUMAN_LINK_TAIL, .6)], sp, r))

    # chat bursts: a human replying quickly in a thread (legit, fast, irregular)
    for i in range(S(20)):
        t = BASE_TS + rng.uniform(0, DAY)
        for _ in range(rng.randint(5, 8)):
            ev.append(dict(text=_fill(_pick(CHAT_BURST, split, rng), rng), actor=f"human_chat_burst-{i}", ts=t,
                           meta=_human_meta(rng, fill=(3, 14)), context=None, category="human_chat_burst"))
            t += rng.lognormvariate(2.9, 0.9)  # median ~18s, heavy tail

    # ---- spam
    for i in range(S(30)):  # AI outreach: single spammer, several sends
        t = BASE_TS + rng.uniform(0, DAY)
        stealthy = rng.random() < 0.3
        for _ in range(rng.randint(3, 7)):
            meta = _bot_meta(rng, ua_bot=0.2 if stealthy else 0.5, fill=(6, 25) if stealthy else (0.3, 4))
            ev.append(dict(text=_compose([(AI_OUTREACH_OPEN, 1), (AI_OUTREACH_BODY, 1), (AI_OUTREACH_BODY, .6), (AI_OUTREACH_CTA, 1)], split, rng),
                           actor=f"ai_outreach-{i}", ts=t, meta=meta, context=None, category="ai_outreach"))
            t += rng.uniform(20, 400)
    for i in range(S(60)):
        ctx = _parent(rng)
        ev.append(dict(text=_compose([(AI_COMMENT_GENERIC, 1), (AI_COMMENT_PLUG, 1)], split, rng), actor=f"ai_comment-{i // 2}",
                       ts=BASE_TS + rng.uniform(0, DAY), meta=_bot_meta(rng, 0.3, (1, 12)), context=ctx, category="ai_comment"))
    for i in range(S(25)):
        t = BASE_TS + rng.uniform(0, DAY)
        for _ in range(4):
            ev.append(dict(text=_compose([(AI_SEO_OPEN, 1), (AI_SEO_BODY, 1), (AI_SEO_CTA, 1)], split, rng), actor=f"ai_seo-{i}", ts=t,
                           meta=_bot_meta(rng, 0.4, (0.5, 8)), context=None, category="ai_seo"))
            t += rng.uniform(30, 900)
    for i in range(S(120)):
        ev.append(dict(text=_fill(_pick(CLASSIC, split, rng), rng), actor=f"classic_spam-{i}", ts=BASE_TS + rng.uniform(0, DAY),
                       meta=_bot_meta(rng, 0.4, (0.5, 10)), context=None, category="classic_spam"))
    for i in range(S(30)):  # stealth agent: fluent, on-topic text; only cadence/metadata give it away
        t = BASE_TS + rng.uniform(0, DAY)
        jitter = rng.random() < 0.5
        meta_base = _bot_meta(rng, ua_bot=0.55, fill=(0.3, 2.5))
        for _ in range(rng.randint(7, 12)):
            ev.append(dict(text=_fill(_pick(STEALTH, split, rng), rng), actor=f"agent_stealth-{i}", ts=t,
                           meta=dict(meta_base, fill_time_s=rng.uniform(0.3, 2.5)), context=None, category="agent_stealth"))
            t += rng.uniform(30, 90) if jitter else 60.0 + rng.uniform(-1, 1)
    for i in range(S(120)):  # paraphrased campaign through many "organic-looking" accounts
        ev.append(dict(text=_fill(_pick(CAMPAIGN_OPEN, split, rng) + " " + _pick(CAMPAIGN_CTA, split, rng), rng), actor=f"paraphrase_campaign-{i}",
                       ts=BASE_TS + rng.uniform(0, DAY), meta=_human_meta(rng, fill=(6, 30)) | {"account_age_s": rng.uniform(3600, 5 * DAY)},
                       context=None, category="paraphrase_campaign"))

    ev.sort(key=lambda e: e["ts"])
    for i, e in enumerate(ev):
        e["id"] = f"{split}-{i}"
        e["label"] = int(e["category"] in SPAM)
        e["text_label"] = None if e["category"] in TEXT_STEALTH else int(e["category"] in TEXT_SPAM)
    return ev


def _praise(rng, split, n):
    out = []
    for i in range(n):
        ctx = _parent(rng)
        topic = re.search(r"into (.+?):", ctx).group(1)
        if rng.random() < 0.4:
            text = _pick(PRAISE_GENERIC_HUMAN, split, rng)
        else:
            # fill the topic slot with the parent's own topic so the reply really engages with it
            text = _pick(PRAISE_SPECIFIC_HUMAN, split, rng).replace("{topic}", topic)
            text = _fill(text, rng)
        out.append(dict(text=text, actor=f"human_praise-{i}", ts=BASE_TS + rng.uniform(0, DAY), meta=_human_meta(rng),
                        context=ctx, category="human_praise"))
    return out
