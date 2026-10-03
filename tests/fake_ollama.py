"""Minimal fake of Ollama's /api/chat and /api/tags, enough to exercise the HTTP path and the pipeline."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _extract(user: str) -> dict:
    sec = re.search(r"Section: (.*)", user).group(1)
    text = user.split("Text:\n", 1)[1]
    money = re.search(r"\$(\d+)", text)
    if "fee" in sec.lower():
        fee_usd = "80" if money and money.group(1) == "8" else (money.group(1) if money else "")  # "80" = deliberate hallucination
        return dict(summary="Apply with proof of rabies shot and pay the fee.", role="procedure", refs=[], steps=[
            dict(actor="Dog owner", action="Submit application with rabies proof", condition="", deadline="", fee=f"${money.group(1)}" if money else "",
                 penalty="", fee_usd=fee_usd, deadline_days="", penalty_max_usd="", modality="obligation", renewal="annual" if "annual" in text else "unspecified"),
            dict(actor="Treasurer", action="Issue license", condition="application complete", deadline="", fee="", penalty="",
                 fee_usd="", deadline_days="", penalty_max_usd="", modality="power", renewal="unspecified")])
    if "licens" in sec.lower() and ("required" in sec.lower() or "Licensing" in sec or "must be" in sec):
        return dict(summary="Dogs over four months must be licensed.", role="requirement", refs=["4-2"], steps=[
            dict(actor="Dog owner", action="Obtain dog license", condition="dog older than four months", deadline="within 30 days" if "30 days" in text else "",
                 fee="", penalty="", fee_usd="", deadline_days="30" if "30 days" in text else "", penalty_max_usd="", modality="obligation",
                 renewal="annual" if "annual" in text else "unspecified")])
    return dict(summary="Unlicensed owners can be penalised.", role="penalty", refs=[], steps=[
        dict(actor="Court", action="Fine unlicensed owner", condition="no license", deadline="", fee="", penalty=f"up to ${money.group(1)}" if money else "misdemeanor",
             fee_usd="", deadline_days="", penalty_max_usd=money.group(1) if money else "", modality="power", renewal="unspecified")])


def _compile(user: str) -> dict:
    lines = [l for l in user.splitlines() if re.match(r"^c\d+\.\d+ \|", l)]
    code = lambda needle: next((l.split(" | ")[0] for l in lines if needle in l), None)
    first, appl, issue, fine = code("Obtain"), code("Submit"), code("Issue license"), code("Fine")
    pick = lambda c: [c] if c else []
    return dict(title="Dog licensing", nodes=[
        dict(id="n1", type="task", actor="Dog owner", label="Obtain dog license", sources=pick(first) + ["c99.1"]),
        dict(id="n2", type="task", actor="Dog owner", label="Submit application with rabies proof", sources=pick(appl)),
        dict(id="n3", type="task", actor="Treasurer", label="Issue license", sources=pick(issue)),
        dict(id="n4", type="gateway", actor="", label="Licensed within deadline?", sources=[]),
        dict(id="n5", type="task", actor="Court", label="Fine owner", sources=pick(fine)),
        dict(id="n6", type="task", actor="Clerk", label="Made-up step", sources=[]),
        dict(id="n7", type="end", actor="", label="Licensed", sources=[]),
        dict(id="n8", type="end", actor="", label="Orphan end", sources=[])],
        edges=[dict(**{"from": "n1", "to": "n2"}), dict(**{"from": "n2", "to": "n3"}), dict(**{"from": "n3", "to": "n4"}),
               dict(**{"from": "n4", "to": "n7", "label": "Yes"}), dict(**{"from": "n4", "to": "n5", "label": "No"}),
               dict(**{"from": "n5", "to": "n2", "label": "Retry"}),
               dict(**{"from": "n1", "to": "zzz"}), dict(**{"from": "n6", "to": "n6"})])


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user, props = body["messages"][1]["content"], body["format"]["properties"]
        out = _extract(user) if "steps" in props and "summary" in props else _compile(user)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"message": {"role": "assistant", "content": json.dumps(out)}}).encode())


def start(port: int = 0):
    srv = HTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
