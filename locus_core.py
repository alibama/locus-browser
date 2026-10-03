"""
locus_core — no Streamlit, no LLM calls. Helpers + the SQLite store everything else writes to.

Everything the pipeline learns is stored, keyed by content hash + model + schema version, so:
  * re-runs are free, * two models never overwrite each other, * you can export it all for review.

Config is read lazily from env so tests can redirect it:
    LOCUS_DB      SQLite file (default ./locus_process.db)
    OLLAMA_HOST   default http://localhost:11434
    OLLAMA_MODEL  default qwen2.5:7b
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

SCHEMA_VER = "v3"
STATUSES = ["llm-draft", "reviewed-ok", "needs-work", "rejected"]
DIMS = {
    "opacity": "Opacity — high = harder for an ordinary person to know what's required",
    "enforcement_discretion": "Enforcement discretion — high = officials choose whether/whom to act against",
    "paternalism": "Paternalism — high = protects the actor from themself; low = protects others",
    "problem_salience": "Problem salience — high = framed as important, urgent or threatening",
}
DIM_LIST = list(DIMS)
_XML_BAD = re.compile("[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]")

# Controlled subject vocabulary. Slugs are what triage tags chunks with and what becomes a
# Wikibase "regulated subject" item; unknown subjects arrive as proposed ('new:<slug>') and wait for approval.
SEED_SUBJECTS = {
    "business-license": "Business license",
    "business-license-tax": "Business license tax",
    "business-tangible-property-tax": "Business tangible personal property tax",
    "zoning-approval": "Zoning approval",
    "home-occupation": "Home occupation permit",
    "sign-permit": "Sign permit",
    "transient-occupancy-tax": "Transient occupancy tax",
    "short-term-rental": "Short-term rental",
    "dog-license": "Dog license",
    "animal-control": "Animal control",
    "building-permit": "Building permit",
    "alcohol-license": "Alcohol license",
    "noise": "Noise",
    "vehicle-storage": "Vehicle storage",
    "food-establishment-permit": "Food establishment permit",
    "fictitious-name": "Fictitious / assumed name registration",
}


def db_path() -> Path:
    return Path(os.environ.get("LOCUS_DB", "locus_process.db"))


def default_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


# ───────────────────────────── small helpers ─────────────────────────────
def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return _XML_BAD.sub("", str(v))


def label(header, n: int = 62) -> str:
    h = re.sub(r"^#+\s*", "", clean(header))
    h = re.sub(r"\s+", " ", h).strip()
    return h if len(h) <= n else h[: n - 1] + "…"


def sid(prefix: str, *parts) -> str:
    h = hashlib.sha1("|".join(clean(p) for p in parts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


def section_no(header) -> str:
    h = re.sub(r"^#+\s*", "", clean(header)).strip()
    m = re.match(r"(?:sec(?:tion|\.)?\s*)?§?\s*(\d+[A-Za-z]?(?:[.\-]\d+[A-Za-z]?)*)", h, re.I)
    return m.group(1) if m else ""


def sec_of(row) -> str:
    """Explicit section number if the source has one, else parse it from the header."""
    s = clean(row.get("section_no") if hasattr(row, "get") else "").strip()
    return s or section_no(row.get("header") if hasattr(row, "get") else "")


def chunk_key(state, place, header, content) -> str:
    return hashlib.sha1("|".join(clean(x) for x in (state, place, header, content)).encode()).hexdigest()[:16]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", clean(s)).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    """Is the quote (whitespace/case-insensitive) literally in the text?"""
    q = norm_ws(quote).strip(" .…\"'“”")
    return len(q) >= 8 and q in norm_ws(text)


def grounded_num(val, text: str) -> str:
    """Keep a model-supplied number only if that number literally appears in the text."""
    v = re.sub(r"[^\d.]", "", clean(val))
    if not v or v == ".":
        return ""
    try:
        n = float(v)
    except ValueError:
        return ""
    t = re.sub(r"(?<=\d),(?=\d)", "", clean(text))
    t = re.sub(r"(?<![\d])\.(\d)", r"0.\1", t)                      # ".36" -> "0.36"
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if any(abs(float(d) - n) < 1e-9 for d in nums):
        return v
    if n < 1 and re.search(rf"\b{round(n * 100)}\s*(?:cents?|¢)", t, re.I):   # 36 cents -> 0.36
        return v
    return ""


# ───────────────────────────── SQLite store ─────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS markup(
    chunk_key TEXT, pass TEXT, model TEXT, schema_ver TEXT, data TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_key, pass, model, schema_ver));
CREATE TABLE IF NOT EXISTS relevance(
    chunk_key TEXT, query_key TEXT, model TEXT, schema_ver TEXT, data TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_key, query_key, model, schema_ver));
CREATE TABLE IF NOT EXISTS query_interp(
    query_key TEXT, model TEXT, schema_ver TEXT, data TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (query_key, model, schema_ver));
CREATE TABLE IF NOT EXISTS regimes(
    rkey TEXT PRIMARY KEY, state TEXT, place TEXT, query TEXT, model TEXT, schema_ver TEXT, data TEXT,
    created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS processes(
    pkey TEXT PRIMARY KEY, state TEXT, place TEXT, query TEXT, model TEXT, prompt_ver TEXT, data TEXT,
    status TEXT DEFAULT 'llm-draft', note TEXT DEFAULT '', reviewer TEXT DEFAULT '',
    updated TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS subjects(
    slug TEXT PRIMARY KEY, label TEXT, status TEXT DEFAULT 'seed', qid TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS llm_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP, stage TEXT, model TEXT, ctx TEXT,
    system TEXT, user TEXT, response TEXT, seconds REAL, error TEXT);
"""


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(db_path(), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(_SCHEMA)
    for slug_, lab in SEED_SUBJECTS.items():
        c.execute("INSERT OR IGNORE INTO subjects(slug, label, status) VALUES (?,?, 'seed')", [slug_, lab])
    c.commit()
    return c


def markup_get(keys: list[str], pass_: str, model: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not keys:
        return out
    with closing(_db()) as c:
        for i in range(0, len(keys), 400):
            part = keys[i:i + 400]
            qm = ",".join("?" * len(part))
            for k, d in c.execute(
                f"SELECT chunk_key, data FROM markup WHERE pass=? AND model=? AND schema_ver=? AND chunk_key IN ({qm})",
                [pass_, model, SCHEMA_VER, *part]):
                out[k] = json.loads(d)
    return out


def markup_put(key: str, pass_: str, model: str, data: dict) -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR REPLACE INTO markup(chunk_key, pass, model, schema_ver, data) VALUES (?,?,?,?,?)",
                  [key, pass_, model, SCHEMA_VER, json.dumps(data)])
        c.commit()


def relevance_get(keys: list[str], query_key: str, model: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not keys:
        return out
    with closing(_db()) as c:
        for i in range(0, len(keys), 400):
            part = keys[i:i + 400]
            qm = ",".join("?" * len(part))
            for k, d in c.execute(
                f"SELECT chunk_key, data FROM relevance WHERE query_key=? AND model=? AND schema_ver=? AND chunk_key IN ({qm})",
                [query_key, model, SCHEMA_VER, *part]):
                out[k] = json.loads(d)
    return out


def relevance_put(key: str, query_key: str, model: str, data: dict) -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR REPLACE INTO relevance(chunk_key, query_key, model, schema_ver, data) VALUES (?,?,?,?,?)",
                  [key, query_key, model, SCHEMA_VER, json.dumps(data)])
        c.commit()


def query_get(query_key: str, model: str) -> dict | None:
    with closing(_db()) as c:
        r = c.execute("SELECT data FROM query_interp WHERE query_key=? AND model=? AND schema_ver=?",
                      [query_key, model, SCHEMA_VER]).fetchone()
    return json.loads(r[0]) if r else None


def query_put(query_key: str, model: str, data: dict) -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR REPLACE INTO query_interp(query_key, model, schema_ver, data) VALUES (?,?,?,?)",
                  [query_key, model, SCHEMA_VER, json.dumps(data)])
        c.commit()


def regime_get(rkey: str) -> dict | None:
    with closing(_db()) as c:
        r = c.execute("SELECT data FROM regimes WHERE rkey=?", [rkey]).fetchone()
    return json.loads(r[0]) if r else None


def regime_put(rkey: str, state: str, place: str, query: str, model: str, data: dict) -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR REPLACE INTO regimes(rkey, state, place, query, model, schema_ver, data) VALUES (?,?,?,?,?,?,?)",
                  [rkey, state, place, query, model, SCHEMA_VER, json.dumps(data)])
        c.commit()


def get_process(pkey: str) -> dict | None:
    with closing(_db()) as c:
        r = c.execute("SELECT data, status, note, reviewer FROM processes WHERE pkey=?", [pkey]).fetchone()
    return None if r is None else {**json.loads(r[0]), "status": r[1], "note": r[2], "reviewer": r[3], "pkey": pkey}


def list_processes() -> list[dict]:
    with closing(_db()) as c:
        rows = c.execute("SELECT pkey, state, place, query, model, status, updated FROM processes ORDER BY updated DESC").fetchall()
    return [dict(zip(("pkey", "state", "place", "query", "model", "status", "updated"), r)) for r in rows]


def save_process(pkey, state, place, query, model, data: dict) -> None:
    with closing(_db()) as c:
        c.execute(
            """INSERT INTO processes(pkey, state, place, query, model, prompt_ver, data) VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(pkey) DO UPDATE SET data=excluded.data, updated=CURRENT_TIMESTAMP""",
            [pkey, state, place, query, model, SCHEMA_VER, json.dumps(data)])
        c.commit()


def patch_process(pkey: str, patch: dict) -> None:
    with closing(_db()) as c:
        r = c.execute("SELECT data FROM processes WHERE pkey=?", [pkey]).fetchone()
        if r:
            d = json.loads(r[0])
            d.update(patch)
            c.execute("UPDATE processes SET data=?, updated=CURRENT_TIMESTAMP WHERE pkey=?", [json.dumps(d), pkey])
            c.commit()


def set_review(pkey: str, status: str, note: str, reviewer: str = "") -> None:
    with closing(_db()) as c:
        c.execute("UPDATE processes SET status=?, note=?, reviewer=?, updated=CURRENT_TIMESTAMP WHERE pkey=?",
                  [status, note, reviewer, pkey])
        c.commit()


def subjects(include_proposed: bool = True) -> dict[str, dict]:
    with closing(_db()) as c:
        rows = c.execute("SELECT slug, label, status, qid FROM subjects ORDER BY slug").fetchall()
    return {s: {"label": l, "status": st, "qid": q} for s, l, st, q in rows if include_proposed or st != "proposed"}


def subject_add(slug_: str, label_: str, status: str = "proposed") -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR IGNORE INTO subjects(slug, label, status) VALUES (?,?,?)", [slug(slug_), label_ or slug_, status])
        c.commit()


def subject_set(slug_: str, status: str | None = None, qid: str | None = None, label_: str | None = None) -> None:
    with closing(_db()) as c:
        if status:
            c.execute("UPDATE subjects SET status=? WHERE slug=?", [status, slug_])
        if qid is not None:
            c.execute("UPDATE subjects SET qid=? WHERE slug=?", [qid, slug_])
        if label_:
            c.execute("UPDATE subjects SET label=? WHERE slug=?", [label_, slug_])
        c.commit()


def log_llm(stage: str, model: str, ctx: str, system: str, user: str, response: str, seconds: float, error: str = "") -> None:
    try:
        with closing(_db()) as c:
            c.execute("INSERT INTO llm_log(stage, model, ctx, system, user, response, seconds, error) VALUES (?,?,?,?,?,?,?,?)",
                      [stage, model, ctx, system, user, response, seconds, error])
            c.commit()
    except sqlite3.Error:
        pass  # logging must never break a run


def llm_stats(model: str) -> dict[str, tuple[float, int]]:
    """stage -> (mean seconds, number of calls) from earlier successful calls on this machine."""
    with closing(_db()) as c:
        rows = c.execute("SELECT stage, avg(seconds), count(*) FROM llm_log WHERE model=? AND (error IS NULL OR error='') "
                         "GROUP BY stage", [model]).fetchall()
    return {s: (a, n) for s, a, n in rows}


def export_debug(pkey: str) -> dict:
    """Everything needed to understand (or reproduce) one process — what to send back when something looks wrong."""
    rec = get_process(pkey)
    if rec is None:
        return {}
    keys = [c["key"] for c in rec.get("chunks", [])]
    model = rec.get("model", "")
    with closing(_db()) as c:
        qm = ",".join("?" * len(keys)) or "''"
        # query / regime / compile calls are logged under the query key, the rest under chunk keys or the process key
        logs = c.execute(
            f"SELECT ts, stage, model, ctx, system, user, response, seconds, error FROM llm_log "
            f"WHERE model IN (?, ?) AND (ctx=? OR ctx=? OR ctx IN ({qm})) ORDER BY id",
            [model, rec.get("verifier") or model, pkey, rec.get("query_key", "-"), *keys]).fetchall()
    return {
        "process": rec, "schema_ver": SCHEMA_VER,
        "triage": markup_get(keys, "triage", model), "extract": markup_get(keys, "extract", model),
        "relevance": relevance_get(keys, rec.get("query_key", ""), model),
        "llm_log": [dict(zip(("ts", "stage", "model", "ctx", "system", "user", "response", "seconds", "error"), r)) for r in logs],
    }


# ───────────────────────────── candidates ─────────────────────────────
def _terms(query: str) -> list[str]:
    return [t for t in re.split(r"\s+", query.lower().strip()) if t]


def score_frame(df: pd.DataFrame, query: str) -> pd.DataFrame:
    terms = _terms(query)
    head = df["header"].fillna("").str.lower()
    hay = head + " " + df["content"].fillna("").str.lower()
    mask = pd.Series(True, index=df.index)
    for t in terms:
        mask &= hay.str.contains(re.escape(t))
    out = df[mask].copy()
    out["_score"] = sum(head[out.index].str.contains(re.escape(t)).astype(int) for t in terms) if terms else 0
    return out


def build_candidates(frames: dict[str, pd.DataFrame], query: str, caps: dict[str, int], fns: list[str],
                     strict: bool = True) -> pd.DataFrame:
    """frames: level -> unified frame (already filtered/ranked for state/federal). Adds ckey + code.
    strict: a local section must have a query word in its title, or contain the whole topic phrase in its text.
    (Otherwise 'business licen' also pulls in 'Fixing payday', which merely mentions both words somewhere.)"""
    parts = []
    for level in ("local", "state", "federal"):
        df = frames.get(level)
        if df is None or df.empty or caps.get(level, 0) <= 0:
            continue
        if level == "local":
            m = score_frame(df, query)
            m = m[m["fn"].isin(fns)]
            if strict and _terms(query):
                phrase = " ".join(_terms(query))
                in_text = m["content"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.contains(re.escape(phrase))
                m = m[(m["_score"] > 0) | in_text]
            m = m.sort_values("_score", ascending=False, kind="stable").head(caps[level]).sort_index().drop(columns="_score")
        else:
            m = df.head(caps[level]).copy()
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["header", "content", "level", "ckey", "code", "place", "state"])
    m = pd.concat(parts, ignore_index=True)
    m["place"] = m["place"].fillna("")
    m["ckey"] = [chunk_key(r.state, r.place, r.header, r.content) for r in m.itertuples()]
    m["code"] = [f"c{i + 1}" for i in range(len(m))]
    return m


def summarize(m: pd.DataFrame, extracts: dict[str, dict], model: str = "") -> dict:
    """One column of the compare table from stored extractions (no LLM)."""
    rules, rows = [], []
    for k in m["ckey"]:
        e = extracts.get(k) or {}
        rules += e.get("rules", [])
        rows += e.get("rows", [])
    done = bool(extracts)

    def join(vals, fmt=str):
        u = []
        for v in vals:
            if v and v not in u:
                u.append(v)
        return "; ".join(fmt(v) for v in u) if u else ("—" if done else "(not run)")

    flat = [r["amount_value"] for r in rules if r.get("amount_kind") == "flat_fee"] + \
           [r["amount_value"] for r in rows if r.get("amount_kind") == "flat_fee"]
    pct = [r["amount_value"] for r in rules if r.get("amount_kind") == "percent_of_base"] + \
          [r["amount_value"] for r in rows if r.get("amount_kind") == "percent_of_base"]
    thr = [f"{r['threshold_variable']} {r['threshold_op']} {r['threshold_value']}".strip() for r in rules if r.get("threshold_value")]
    return {
        "Model": model or "—",
        "Sections": len(m),
        "…local / state / federal": " / ".join(str(int((m["level"] == l).sum())) for l in ("local", "state", "federal")) if len(m) else "0 / 0 / 0",
        "Flat fee (USD)": join(flat, lambda v: f"${v}"),
        "Rate (% of base)": join(pct, lambda v: f"{v}%"),
        "Thresholds": join(thr),
        "Deadline (days)": join([r.get("deadline_days") for r in rules]),
        "Fixed due date": join([r.get("deadline_fixed") for r in rules]),
        "Renewal": join([r.get("renewal") for r in rules if r.get("renewal") not in (None, "", "unspecified")]),
        "Max penalty": join([f"{r['penalty_max_value']} {r.get('penalty_unit', '')}".strip() for r in rules if r.get("penalty_max_value")]),
        "Fees as written": join([r.get("amount_text") for r in rules]),
    }
