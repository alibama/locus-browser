"""
locus_hf — Hugging Face access from inside the app (no environment variables needed; works on Windows).

Token sources, first match wins:
    1. the token typed into the app this session
    2. HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN
    3. a token this app saved on this computer (~/.locus_explorer/hf_token), only if you ticked "remember"
    4. the Hugging Face CLI's own token file (~/.cache/huggingface/token or $HF_HOME/token) — i.e. you ran `huggingface-cli login`

The app never writes to the Hugging Face CLI's token file, and never writes a token anywhere unless asked.
`HF_ENDPOINT` (the standard huggingface_hub variable) overrides https://huggingface.co.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

OUL_REPO = "vaquill/open-us-law"
ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def own_token_path() -> Path:
    return Path.home() / ".locus_explorer" / "hf_token"


def cli_token_path() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "token"


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def find_token(session_token: str = "") -> tuple[str, str]:
    """-> (token, source) where source is 'typed in app' | 'environment' | 'saved by this app' | 'huggingface-cli login' | ''."""
    if session_token.strip():
        return session_token.strip(), "typed in app"
    for n in ENV_NAMES:
        if os.environ.get(n, "").strip():
            return os.environ[n].strip(), f"environment ({n})"
    t = _read(own_token_path())
    if t:
        return t, "saved by this app"
    t = _read(cli_token_path())
    if t:
        return t, "huggingface-cli login"
    return "", ""


def save_token(token: str) -> Path:
    p = own_token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token.strip(), encoding="utf-8")
    try:
        os.chmod(p, 0o600)          # no-op on Windows; the file sits in your own profile folder
    except OSError:
        pass
    return p


def forget_saved_token() -> bool:
    p = own_token_path()
    if p.exists():
        p.unlink()
        return True
    return False


def fingerprint(token: str) -> str:
    return hashlib.sha1(token.encode()).hexdigest()[:8] if token else ""


def mask(token: str) -> str:
    return "" if not token else (token[:3] + "…" + token[-4:] if len(token) > 10 else "…")


def looks_like_token(token: str) -> bool:
    return token.startswith("hf_") and len(token) >= 20


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):     # a 3xx means "authorised"; never forward the token to the CDN host
        return None


def _call(url: str, token: str, method: str = "GET", timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {token}"} if token else {})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read(20000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(2000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return 0, str(getattr(e, "reason", e))


def whoami(token: str) -> dict:
    """{'ok': bool, 'name': str, 'error': str}"""
    if not token:
        return {"ok": False, "name": "", "error": "No token."}
    code, body = _call(f"{endpoint()}/api/whoami-v2", token)
    if code == 200:
        try:
            return {"ok": True, "name": json.loads(body).get("name", "?"), "error": ""}
        except json.JSONDecodeError:
            return {"ok": True, "name": "?", "error": ""}
    if code == 401:
        return {"ok": False, "name": "", "error": "Hugging Face rejected this token (401). Check it was copied completely."}
    if code == 0:
        return {"ok": False, "name": "", "error": f"Could not reach Hugging Face: {body}"}
    return {"ok": False, "name": "", "error": f"Unexpected answer from Hugging Face (HTTP {code})."}


def dataset_access(token: str, repo: str = OUL_REPO, probe: str = "totals.json") -> dict:
    """Can this token read the dataset's files? state: ok | no-token | bad-token | gated | missing | network"""
    if not token:
        return {"state": "no-token", "message": "No token. The dataset is gated, so a token is required."}
    code, _ = _call(f"{endpoint()}/datasets/{repo}/resolve/main/{probe}", token, "HEAD")
    if code == 200 or 300 <= code < 400:
        return {"state": "ok", "message": f"Access to {repo} confirmed."}
    if code == 401:
        return {"state": "bad-token", "message": "Hugging Face rejected the token (401)."}
    if code == 403:
        return {"state": "gated", "message":
                f"Token is valid but cannot read {repo}. Open the dataset page while logged in and accept its conditions. "
                "If you made a fine-grained token, it also needs permission to read public gated repos; a classic 'Read' token is simplest."}
    if code == 404:
        return {"state": "missing", "message": f"Could not find {probe} in {repo} (HTTP 404); the repo layout may have changed."}
    if code == 0:
        return {"state": "network", "message": "Could not reach Hugging Face."}
    return {"state": "missing", "message": f"Unexpected HTTP {code}."}


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def apply_to_duckdb(con, token: str) -> tuple[bool, str]:
    """Load httpfs and register the token as a DuckDB secret. Reports errors instead of hiding them."""
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    except Exception as e:  # noqa: BLE001
        return False, (f"DuckDB's httpfs extension could not be loaded ({str(e).splitlines()[0][:160]}). "
                       "It downloads once from extensions.duckdb.org, so it needs internet access (and a proxy setting if your network uses one).")
    if not token:
        return True, "httpfs loaded (no token)."
    try:
        con.execute(f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN {_lit(token)})")
    except Exception as e:  # noqa: BLE001
        return False, f"Could not register the token with DuckDB: {str(e).splitlines()[0][:200]}"
    return True, "Token registered with DuckDB."


def probe_duckdb(cur, template: str, jur: str = "va", corpus: str = "statutes") -> dict:
    """The real end-to-end test: DuckDB reads a file's footer through hf:// with the registered token."""
    path = template.format(jur=jur, corpus=corpus)
    try:
        n = cur.execute(f"SELECT count(*) FROM read_parquet({_lit(path)})").fetchone()[0]
        return {"ok": True, "rows": int(n), "error": "", "path": path}
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:240]
        hint = ""
        if "403" in msg or "401" in msg:
            hint = " (access denied: token missing, or the dataset's conditions are not accepted)"
        elif "404" in msg or "No files found" in msg:
            hint = " (file not found: check the file template and the jurisdiction code)"
        return {"ok": False, "rows": 0, "error": msg + hint, "path": path}
