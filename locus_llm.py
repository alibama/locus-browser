"""
locus_llm — Ollama client + model-server management (status, start, pull, self-test).

Every chat call is written to llm_log (prompt, raw response, seconds, error) so a bad result can be
inspected and sent back for debugging.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import locus_core as C


class LLMUnavailable(RuntimeError):
    pass


def host_url(h: str) -> str:
    h = (h or "").strip().rstrip("/")
    return h if h.startswith("http") else f"http://{h}"


def server_status(host: str, timeout: float = 3.0) -> dict:
    """{'up': bool, 'models': [{'name','size'}], 'error': str}"""
    try:
        with urllib.request.urlopen(host_url(host) + "/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return {"up": True, "models": [{"name": m["name"], "size": m.get("size", 0)} for m in data.get("models", [])], "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"up": False, "models": [], "error": str(getattr(e, "reason", e))}


def model_available(status: dict, model: str) -> bool:
    names = {m["name"] for m in status.get("models", [])}
    return model in names or (":" not in model and f"{model}:latest" in names)


def start_server(host: str, wait: float = 12.0) -> tuple[bool, str]:
    """Start `ollama serve` locally. Only works if the binary is on PATH and the host is this machine."""
    if not any(h in host_url(host) for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return False, "Host is not local; start Ollama on that machine."
    exe = shutil.which("ollama")
    if not exe:
        return False, "`ollama` not found on PATH. Install it from https://ollama.com, then retry."
    kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":      # start_new_session is POSIX-only; detach properly on Windows
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    subprocess.Popen([exe, "serve"], **kw)
    t0 = time.time()
    while time.time() - t0 < wait:
        if server_status(host, 1.0)["up"]:
            return True, "Ollama is up."
        time.sleep(0.5)
    return False, "Started `ollama serve` but it did not answer in time; check the terminal/logs."


def pull_model(host: str, model: str):
    """Generator of progress dicts {'status','completed','total'} from /api/pull (streaming)."""
    body = json.dumps({"name": model, "model": model, "stream": True}).encode()
    req = urllib.request.Request(host_url(host) + "/api/pull", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d.get("error"):
                        raise LLMUnavailable(d["error"])
                    yield d
    except urllib.error.URLError as e:
        raise LLMUnavailable(f"Cannot reach Ollama at {host}: {e.reason}") from e


def call_llm(model: str, host: str, system: str, user: str, schema: dict, *, stage: str = "", ctx: str = "",
             timeout: int = 600) -> dict:
    """Ollama /api/chat with a JSON-schema `format`. temperature 0 + fixed seed for repeatability."""
    body = {
        "model": model, "stream": False, "format": schema,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": 7, "num_ctx": 8192},
    }
    req = urllib.request.Request(host_url(host) + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, raw, err, last = time.time(), "", "", None
    try:
        for _ in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                raw = data["message"]["content"]
                return json.loads(raw)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code == 404:
                    raise LLMUnavailable(f"Model '{model}' is not available on {host} (HTTP 404). Pull it from the sidebar. {msg}") from e
                raise LLMUnavailable(f"Ollama error {e.code}: {msg}") from e
            except urllib.error.URLError as e:
                raise LLMUnavailable(f"Cannot reach Ollama at {host}: {e.reason}") from e
            except (json.JSONDecodeError, KeyError) as e:
                last = e
        raise ValueError(f"Model returned unusable JSON: {last}")
    except Exception as e:
        err = str(e)
        raise
    finally:
        C.log_llm(stage, model, ctx, system, user, raw, time.time() - t0, err)


_TEST_SCHEMA = {"type": "object", "properties": {"fee": {"type": "string"}, "days": {"type": "string"}},
                "required": ["fee", "days"]}


def model_selftest(host: str, model: str) -> dict:
    """Can this model follow a JSON schema on a trivial extraction? -> {'ok','seconds','error','output'}"""
    t0 = time.time()
    try:
        out = call_llm(model, host, "Extract the fee in dollars and the filing deadline in days as digits only.",
                       "The fee is $10 per dog. File within 30 days of acquiring the dog.", _TEST_SCHEMA, stage="selftest", timeout=300)
        ok = "10" in str(out.get("fee", "")) and "30" in str(out.get("days", ""))
        return {"ok": ok, "seconds": round(time.time() - t0, 1), "error": "" if ok else "Answered, but the values were wrong.", "output": out}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 1), "error": str(e), "output": None}
