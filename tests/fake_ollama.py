"""
A fake Ollama (/api/tags, /api/pull streaming, /api/chat with JSON-schema `format`) that is just smart enough to
drive the whole markup pipeline on the synthetic data. It also records concurrency, so tests can prove two models
really were in flight at the same time.
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"installed": {"qwen2.5:7b", "qwen2.5:14b", "tiny", "qwen3:14b", "reason-weird"}, "calls": [], "inflight": {},
         "max_models_inflight": 0, "max_inflight": 0, "delay": 0.03, "bodies": [], "junk_goal": False}


def reset():
    with LOCK:
        STATE.update(installed={"qwen2.5:7b", "qwen2.5:14b", "tiny", "qwen3:14b", "reason-weird"}, calls=[], inflight={},
                     max_models_inflight=0, max_inflight=0, bodies=[], junk_goal=False)


def _quote(text: str, needle: str) -> str:
    i = text.find(needle)
    return text[i:i + len(needle)] if i >= 0 else ""


def triage(sec: str, text: str) -> dict:
    s = sec.lower()
    d = dict(role="other", audience="general", subjects=[], regime="", summary="", self_contained=True, refs=[])
    if "dog license required" in s or "licensing of dogs" in s:
        d.update(role="duty", audience="regulated_person", subjects=["dog-license"], regime="dog licensing", summary="Dogs over four months must be licensed.", refs=[dict(target="4-2", relation="procedure_in")])
    elif "license fee" in s or s.strip().endswith("fee"):
        d.update(role="procedure", audience="regulated_person", subjects=["dog-license"], regime="dog licensing", summary="Apply with proof of rabies shot and pay the fee.")
    elif "failure to license" in s or "penalty" in s:
        d.update(role="penalty", audience="regulated_person", subjects=["dog-license"], regime="dog licensing", summary="Unlicensed owners can be penalised.")
    elif "business license required" in s:
        d.update(role="duty", audience="regulated_person", subjects=["business-license"], regime="business license tax", summary="You need a city business license before doing business.")
    elif "license tax rates" in s:
        d.update(role="rate_schedule", audience="regulated_person", subjects=["business-license-tax"], regime="business license tax", summary="Tax rates by business class.")
    elif "inspector" in s:
        d.update(role="admin_power", audience="official", subjects=["business-license"], regime="business license tax", summary="The commissioner appoints a license inspector.")
    elif "inoperable" in s:
        d.update(role="duty", audience="regulated_person", subjects=["vehicle-storage"], regime="vehicle storage", summary="Do not store broken-down cars in view.")
    elif "local license taxes" in s:
        d.update(role="authority", audience="official", subjects=["business-license-tax", "brand-new-thing"], regime="business license tax", summary="State law limits local license taxes below $100,000 of receipts.")
        d["subjects"] = ["business-license-tax", "new:brand-new-thing"]
    elif "barking" in s:
        d.update(role="duty", audience="regulated_person", subjects=["noise"], regime="noise", summary="No barking at night.")
    return d


def relevance(goal: str, sec: str, text: str) -> dict:
    s, g = sec.lower(), goal.lower()
    if "dog" in g:
        ok = "dog" in s or "dog" in text.lower()
        return dict(verdict="yes" if ok and "barking" not in s else "no", reason="about dog licensing" if ok else "different subject", phase="preconditions" if ok else "out_of_scope")
    if "inspector" in s or "inoperable" in s:
        return dict(verdict="no", reason="internal administration / different subject", phase="out_of_scope")
    if "local license taxes" in s:
        return dict(verdict="partial", reason="state ceiling that constrains the local tax", phase="scope")
    return dict(verdict="yes", reason="part of obtaining a business license", phase="preconditions")


def rules(sec: str, text: str, model: str) -> dict:
    s = sec.lower()
    base = dict(rule_type="duty", actor="", action="", condition="", applies_if=[], deadline_text="", deadline_days="", deadline_fixed="",
                deadline_anchor="", amount_text="", amount_kind="none", amount_value="", amount_unit="", amount_base="",
                threshold_variable="", threshold_op="", threshold_value="", threshold_unit="", penalty_text="", penalty_kind="none",
                penalty_max_value="", penalty_unit="", renewal="unspecified", evidence="")
    mk = lambda **k: {**base, **k}
    if "dog license required" in s or "licensing of dogs" in s:
        has30 = "30 days" in text
        return dict(rules=[mk(rule_type="duty", actor="Dog owner", action="Obtain dog license", condition="dog older than four months",
                              deadline_text="within 30 days" if has30 else "", deadline_days="30" if has30 else "",
                              renewal="annual" if "annual" in text else "unspecified",
                              evidence=_quote(text, "shall obtain a license from the city treasurer within 30 days") or _quote(text, "must be licensed by the county treasurer annually"))])
    if s.strip().endswith("fee") or "license fee" in s:
        m = re.search(r"\$(\d+)", text)
        fee = "80" if m and m.group(1) == "8" else (m.group(1) if m else "")       # "80" = deliberate hallucination
        r = [mk(rule_type="procedure_step", actor="Dog owner", action="Submit application with rabies proof", amount_text=f"${m.group(1)}" if m else "",
                amount_kind="flat_fee" if (fee and model != "tiny") else "none", amount_value=fee if model != "tiny" else "", amount_unit="usd",
                evidence=_quote(text, f"The fee is ${m.group(1)} per dog") or _quote(text, "The annual dog license fee is $8") if m else ""),
             mk(rule_type="procedure_step", actor="Treasurer", action="Issue license", condition="application complete",
                evidence="the treasurer shall promptly issue the license")]                   # fabricated quote -> verifier downgrades to partial
        return dict(rules=r)
    if "failure to license" in s or "penalty" in s:
        m = re.search(r"\$(\d+)", text)
        return dict(rules=[mk(rule_type="penalty", actor="Court", action="Fine unlicensed owner", penalty_text="fine" if m else "Class 4 misdemeanor",
                              penalty_kind="fine" if m else "misdemeanor", penalty_max_value=m.group(1) if m else "", penalty_unit="usd",
                              evidence=_quote(text, f"shall be fined not more than ${m.group(1)}") if m else _quote(text, "Violation is a Class 4 misdemeanor"))])
    if "business license required" in s:
        return dict(rules=[mk(rule_type="duty", actor="Business owner", action="Obtain city business license", deadline_text="within 30 days", deadline_days="30",
                              evidence=_quote(text, "shall obtain a city business license within 30 days"), renewal="annual")])
    if "local license taxes" in s:
        return dict(rules=[mk(rule_type="threshold", actor="", action="", threshold_variable="gross receipts", threshold_op="<", threshold_value="100000",
                              threshold_unit="usd", condition="no local license tax below the threshold", evidence=_quote(text, "gross receipts of less than $100,000"))])
    return dict(rules=[])


def rows(text: str) -> dict:
    return dict(table_title="License tax rates", rows=[
        dict(class_label="Class I", covers="Retail", amount_kind="percent_of_base", amount_value="0.20", amount_unit="usd_per_100", amount_base="gross receipts", min_receipts="100000", max_receipts="", note="", evidence=_quote(text, "Class I: 20 cents per $100")),
        dict(class_label="Class IV", covers="Services", amount_kind="percent_of_base", amount_value="0.36", amount_unit="usd_per_100", amount_base="gross receipts", min_receipts="100000", max_receipts="", note="", evidence=_quote(text, "Class IV: 36 cents per $100")),
        dict(class_label="Any", covers="Small businesses", amount_kind="flat_fee", amount_value="35", amount_unit="usd", amount_base="", min_receipts="", max_receipts="50000", note="", evidence=_quote(text, "flat fee of $35"))])


def compile_(user: str) -> dict:
    lines = [l for l in user.splitlines() if re.match(r"^c\d+\.\d+ \|", l)]
    nodes, edges, prev = [], [], None
    pen = [l for l in lines if "| penalty |" in l]
    main = [l for l in lines if l not in pen and "| threshold |" not in l]
    for i, l in enumerate(main, 1):
        parts = [p.strip() for p in l.split("|")]
        actor = next((p[7:] for p in parts if p.startswith("actor: ")), "")
        action = next((p[8:] for p in parts if p.startswith("action: ")), "step")
        nodes.append(dict(id=f"n{i}", type="task", actor=actor, label=action, sources=[parts[0]], decision=""))
        if prev:
            edges.append({"from": prev, "to": f"n{i}", "label": ""})
        prev = f"n{i}"
        if "DECISION TABLES available" in user and i == 1:
            nodes.append(dict(id="nD", type="task", actor=actor, label="Determine license tax", sources=[parts[0]], decision="D1"))
            edges.append({"from": prev, "to": "nD", "label": ""})
            prev = "nD"
    nodes.append(dict(id="g", type="gateway", actor="", label="Compliant?", sources=[], decision=""))
    nodes.append(dict(id="ok", type="end", actor="", label="Licensed", sources=[], decision=""))
    if prev:
        edges.append({"from": prev, "to": "g", "label": ""})
    edges.append({"from": "g", "to": "ok", "label": "Yes"})
    prev = "g"
    for j, l in enumerate(pen, 1):
        parts = [p.strip() for p in l.split("|")]
        nodes.append(dict(id=f"p{j}", type="task", actor=next((p[7:] for p in parts if p.startswith("actor: ")), ""), label=next((p[8:] for p in parts if p.startswith("action: ")), "penalty"), sources=[parts[0]], decision=""))
        edges.append({"from": prev, "to": f"p{j}", "label": "No" if prev == "g" else ""})
        prev = f"p{j}"
    nodes += [dict(id="x1", type="task", actor="Clerk", label="Made-up step", sources=["c99.1"], decision=""),
              dict(id="x2", type="end", actor="", label="Orphan end", sources=[], decision="")]
    edges += [{"from": "n1", "to": "zzz", "label": ""}, {"from": "x1", "to": "x1", "label": ""}]
    if len(main) >= 1 and pen:
        edges.append({"from": prev, "to": "n1", "label": "Retry"})                    # a loop
    return dict(title="Licensing", nodes=nodes, edges=edges)


def regime(user: str) -> dict:
    phases: dict[str, list] = {}
    excluded = []
    m = {"duty": "preconditions", "procedure": "application", "rate_schedule": "determination", "penalty": "violation_and_penalty", "authority": "scope"}
    for l in user.splitlines():
        mm = re.match(r"^(c\d+) \| ", l)
        if not mm:
            continue
        if "EXCLUDED earlier" in l:
            excluded.append(dict(code=mm.group(1), reason="set aside earlier"))
            continue
        role = re.search(r"role=(\w+)", l).group(1)
        phases.setdefault(m.get(role, "scope"), []).append(mm.group(1))
    return dict(title="Licensing", subject="business-license" if "business" in user.lower() else "dog-license",
                phases=[dict(phase=p, codes=c, note="") for p, c in phases.items()], excluded=excluded,
                missing=[dict(item="renewal due date", why="no section states it")])


def verify(user: str) -> dict:
    text = user.split("CLAIM:", 1)[0].replace("TEXT:\n", "")
    claim = user.split("CLAIM:\n", 1)[1]
    if "Made-up" in claim:
        return dict(verdict="unsupported", issues=["not in text"], quote="")
    first = re.split(r"(?<=[.])\s", text.strip())[0][:120]
    if '"action": "Issue license"' in claim:
        return dict(verdict="supported", issues=[], quote="the treasurer shall promptly issue the license")   # not verbatim in text
    return dict(verdict="supported", issues=[], quote=first)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with LOCK:
            models = [{"name": m, "size": 1234} for m in sorted(STATE["installed"])]
        self._send({"models": models})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/api/pull"):
            self.send_response(200)
            self.end_headers()
            for st, c in (("pulling manifest", 0), ("pulling layer", 50), ("pulling layer", 100)):
                self.wfile.write((json.dumps({"status": st, "completed": c, "total": 100}) + "\n").encode())
            with LOCK:
                STATE["installed"].add(body["name"])
            self.wfile.write((json.dumps({"status": "success"}) + "\n").encode())
            return
        model = body["model"]
        with LOCK:
            known = model in STATE["installed"]
            STATE["bodies"].append({"model": model, "think": body.get("think", "absent"), "num_ctx": body["options"]["num_ctx"]})
        if model == "reason-weird" and "think" in body:
            return self._send({"error": "this model does not support think"}, 400)
        if not known:
            return self._send({"error": f"model '{model}' not found"}, 404)
        with LOCK:
            STATE["inflight"][model] = STATE["inflight"].get(model, 0) + 1
            STATE["max_models_inflight"] = max(STATE["max_models_inflight"], sum(1 for v in STATE["inflight"].values() if v))
            STATE["max_inflight"] = max(STATE["max_inflight"], sum(STATE["inflight"].values()))
        try:
            time.sleep(STATE["delay"])
            user, props = body["messages"][1]["content"], body["format"]["properties"]
            sec = (re.search(r"Section: (.*)", user) or [None, ""])[1] if re.search(r"Section: (.*)", user) else ""
            text = user.split("Text:\n", 1)[1] if "Text:\n" in user else ""
            if "role" in props and "audience" in props:
                stage, out = "triage", triage(sec, text)
            elif "goal" in props and "keywords" in props:
                stage = "query"
                out = dict(goal="search" if STATE["junk_goal"] else ("Obtain and keep a dog license" if "dog" in user.lower() else "Obtain a business license"),
                           audience="regulated_person", subjects=["dog-license"] if "dog" in user.lower() else ["business-license"], keywords=["license"])
            elif "verdict" in props and "phase" in props:
                stage, out = "relevance", relevance(user.split("\n")[0], sec, text)
            elif "verdict" in props and "issues" in props:
                stage, out = "verify", verify(user)
            elif "rules" in props:
                stage, out = "rules", rules(sec, text, model)
            elif "rows" in props:
                stage, out = "rates", rows(text)
            elif "terms" in props:
                stage, out = "defs", dict(terms=[])
            elif "phases" in props:
                stage, out = "regime", regime(user)
            elif "nodes" in props:
                stage, out = "compile", compile_(user)
            else:
                stage, out = "selftest", dict(fee="$10", days="30")
            with LOCK:
                STATE["calls"].append((model, stage))
            self._send({"message": {"role": "assistant", "content": json.dumps(out)}})
        finally:
            with LOCK:
                STATE["inflight"][model] -= 1


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ───────────────────────── fake Hugging Face (whoami + gated resolve) ─────────────────────────
class HFH(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, code, body="", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body.encode())

    def _handle(self):
        auth = self.headers.get("Authorization", "")
        HFH.seen.append((self.command, self.path, auth))
        tok = auth.replace("Bearer ", "")
        if self.path.startswith("/api/whoami-v2"):
            return self._send(200, json.dumps({"name": "alice"})) if tok in ("hf_good_" + "x" * 20, "hf_nogate_" + "x" * 20) else self._send(401, "{}")
        if "/resolve/main/" in self.path:
            if not tok:
                return self._send(401)
            if tok == "hf_good_" + "x" * 20:
                return self._send(302, "", {"Location": "http://127.0.0.1:1/cdn-signed-url"})   # must NOT be followed
            if tok == "hf_nogate_" + "x" * 20:
                return self._send(403)
            return self._send(401)
        self._send(404)

    do_GET = do_HEAD = _handle


def start_hf(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), HFH)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
