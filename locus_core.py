"""
locus_core — everything that is not Streamlit: helpers, SQLite store, Ollama client,
section annotation, process compilation, matching and summaries.

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
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path

import pandas as pd

PROMPT_VER = "v2"
STATUSES = ["llm-draft", "reviewed-ok", "needs-work", "rejected"]
DIMS = {
    "opacity": "Opacity — high = harder for an ordinary person to know what's required",
    "enforcement_discretion": "Enforcement discretion — high = officials choose whether/whom to act against",
    "paternalism": "Paternalism — high = protects the actor from themself; low = protects others",
    "problem_salience": "Problem salience — high = framed as important, urgent or threatening",
}
DIM_LIST = list(DIMS)
MODALITIES = ["obligation", "prohibition", "permission", "power"]
RENEWALS = ["annual", "biennial", "one-time", "none", "unspecified"]
TEXT_KEYS = ("actor", "action", "condition", "deadline", "fee", "penalty")
FACT_KEYS = ("modality", "fee_usd", "deadline_days", "renewal", "penalty_max_usd")

_XML_BAD = re.compile("[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]")


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


def chunk_key(state, place, header, content) -> str:
    return hashlib.sha1("|".join(clean(x) for x in (state, place, header, content)).encode()).hexdigest()[:16]


def grounded_num(val, text: str) -> str:
    """Keep a model-supplied number only if that number literally appears in the text."""
    v = re.sub(r"[^\d.]", "", clean(val))
    if not v:
        return ""
    try:
        n = float(v)
    except ValueError:
        return ""
    nums = re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", clean(text)))
    return v if any(float(d) == n for d in nums) else ""


# ───────────────────────────── storage (SQLite) ─────────────────────────────
def _db() -> sqlite3.Connection:
    c = sqlite3.connect(db_path())
    c.executescript(
        """CREATE TABLE IF NOT EXISTS annotations(
               chunk_key TEXT, model TEXT, prompt_ver TEXT, data TEXT,
               created TEXT DEFAULT CURRENT_TIMESTAMP,
               PRIMARY KEY (chunk_key, model, prompt_ver));
           CREATE TABLE IF NOT EXISTS processes(
               pkey TEXT PRIMARY KEY, state TEXT, place TEXT, query TEXT, model TEXT,
               prompt_ver TEXT, data TEXT, status TEXT DEFAULT 'llm-draft', note TEXT DEFAULT '',
               updated TEXT DEFAULT CURRENT_TIMESTAMP);"""
    )
    return c


def get_annotations(keys: list[str], model: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with closing(_db()) as c:
        for i in range(0, len(keys), 400):
            part = keys[i:i + 400]
            qm = ",".join("?" * len(part))
            for k, d in c.execute(
                f"SELECT chunk_key, data FROM annotations WHERE model=? AND prompt_ver=? AND chunk_key IN ({qm})",
                [model, PROMPT_VER, *part],
            ):
                out[k] = json.loads(d)
    return out


def save_annotation(key: str, model: str, data: dict) -> None:
    with closing(_db()) as c:
        c.execute("INSERT OR REPLACE INTO annotations(chunk_key, model, prompt_ver, data) VALUES (?,?,?,?)",
                  [key, model, PROMPT_VER, json.dumps(data)])
        c.commit()


def get_process(pkey: str) -> dict | None:
    with closing(_db()) as c:
        r = c.execute("SELECT data, status, note FROM processes WHERE pkey=?", [pkey]).fetchone()
    return None if r is None else {**json.loads(r[0]), "status": r[1], "note": r[2], "pkey": pkey}


def list_processes() -> list[dict]:
    with closing(_db()) as c:
        rows = c.execute("SELECT pkey, state, place, query, model, status, updated FROM processes ORDER BY updated DESC").fetchall()
    return [dict(zip(("pkey", "state", "place", "query", "model", "status", "updated"), r)) for r in rows]


def save_process(pkey, state, place, query, model, data: dict) -> None:
    with closing(_db()) as c:
        c.execute(
            """INSERT INTO processes(pkey, state, place, query, model, prompt_ver, data)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(pkey) DO UPDATE SET data=excluded.data, updated=CURRENT_TIMESTAMP""",
            [pkey, state, place, query, model, PROMPT_VER, json.dumps(data)],
        )
        c.commit()


def set_review(pkey: str, status: str, note: str) -> None:
    with closing(_db()) as c:
        c.execute("UPDATE processes SET status=?, note=?, updated=CURRENT_TIMESTAMP WHERE pkey=?", [status, note, pkey])
        c.commit()


# ───────────────────────────── Ollama ─────────────────────────────
class LLMUnavailable(RuntimeError):
    pass


def _host(h: str) -> str:
    h = h.strip().rstrip("/")
    return h if h.startswith("http") else f"http://{h}"


def call_llm(model: str, host: str, system: str, user: str, schema: dict, timeout: int = 600) -> dict:
    """Ollama /api/chat with a JSON-schema `format`. temperature 0 + fixed seed for repeatability."""
    body = {
        "model": model, "stream": False, "format": schema,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": 7, "num_ctx": 8192},
    }
    req = urllib.request.Request(_host(host) + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    last: Exception | None = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            return json.loads(data["message"]["content"])
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"Cannot reach Ollama at {host}: {e.reason}") from e
        except (json.JSONDecodeError, KeyError) as e:
            last = e
    raise ValueError(f"Model returned unusable JSON: {last}")


def ollama_models(host: str) -> list[str]:
    with urllib.request.urlopen(_host(host) + "/api/tags", timeout=5) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


_STR = {"type": "string"}
_STEP_PROPS = {k: _STR for k in (*TEXT_KEYS, "fee_usd", "deadline_days", "penalty_max_usd")}
_STEP_PROPS["modality"] = {"type": "string", "enum": MODALITIES}
_STEP_PROPS["renewal"] = {"type": "string", "enum": RENEWALS}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _STR,
        "role": {"type": "string", "enum": ["requirement", "procedure", "exemption", "penalty", "definition", "other"]},
        "steps": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "properties": _STEP_PROPS, "required": list(_STEP_PROPS)}},
        "refs": {"type": "array", "items": _STR},
    },
    "required": ["summary", "role", "steps", "refs"],
}
COMPILE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR,
        "nodes": {"type": "array", "maxItems": 24, "items": {
            "type": "object",
            "properties": {
                "id": _STR, "type": {"type": "string", "enum": ["task", "gateway", "end"]},
                "actor": _STR, "label": _STR, "sources": {"type": "array", "items": _STR}},
            "required": ["id", "type", "actor", "label", "sources"]}},
        "edges": {"type": "array", "items": {
            "type": "object", "properties": {"from": _STR, "to": _STR, "label": _STR},
            "required": ["from", "to"]}},
    },
    "required": ["title", "nodes", "edges"],
}
EXTRACT_SYSTEM = (
    "You extract structured facts from ONE section of a US municipal or county ordinance. "
    "Use only the text given. If something is not stated, return an empty string — never guess amounts, "
    "deadlines or offices. 'actor' is the person or office that must act or may act, as a short consistent "
    "role name (e.g. 'Dog owner', 'City treasurer', 'Animal control officer', 'Court'). 'action' is an "
    "imperative phrase of at most 12 words. 'summary' is one plain-English sentence a non-lawyer can follow. "
    "'refs' lists other section numbers this section points to. A section with no actionable step "
    "(definitions, purpose) returns an empty steps list. "
    "Typed fields: fee_usd, deadline_days and penalty_max_usd are digits only, copied from the text "
    "(e.g. '10', '30'); leave them empty if the text gives no such number or it is not dollars/days. "
    "'modality': obligation (must/shall), prohibition (shall not/unlawful), permission (may), power (an "
    "office is authorised to act). 'renewal': annual only if the license or permit must be renewed every "
    "year, biennial every two years, one-time if issued once, none if it does not expire, else unspecified."
)
COMPILE_SYSTEM = (
    "You assemble extracted ordinance steps into ONE process model a lawyer can check. Use ONLY the steps "
    "given; never add steps, amounts or deadlines. Node types: 'task' (an actor does something), "
    "'gateway' (a yes/no or multi-way question; label it as a question and label every outgoing edge with "
    "the answer), 'end' (an outcome such as 'License issued' or 'Citation issued'). Every task node must "
    "list the step codes (like c3.1) it comes from in 'sources'. Order the nodes by the logical sequence "
    "implied by deadlines, conditions and cross-references (refs), not by section order. Keep the same actor "
    "name for the same role. Penalty steps belong after a gateway such as 'Complied?'. If the text gives no "
    "basis for connecting two steps, do not invent a connection. At most 18 nodes."
)


def norm_annotation(d: dict, text: str = "") -> dict:
    steps = []
    for s in d.get("steps", []) or []:
        st = {k: clean(s.get(k)).strip() for k in TEXT_KEYS}
        if not st["action"]:
            continue
        st["fee_usd"] = grounded_num(s.get("fee_usd"), f"{st['fee']} {text}")
        st["deadline_days"] = grounded_num(s.get("deadline_days"), f"{st['deadline']} {text}")
        st["penalty_max_usd"] = grounded_num(s.get("penalty_max_usd"), f"{st['penalty']} {text}")
        mod, ren = clean(s.get("modality")), clean(s.get("renewal"))
        st["modality"] = mod if mod in MODALITIES else ""
        st["renewal"] = ren if ren in RENEWALS else "unspecified"
        steps.append(st)
    return {"summary": clean(d.get("summary")).strip(), "role": clean(d.get("role")) or "other",
            "steps": steps, "refs": [clean(r).strip() for r in d.get("refs", []) or []]}


def annotate(rows: pd.DataFrame, model: str, host: str, progress=None) -> list[str]:
    errs: list[str] = []
    n = len(rows)
    for i, (_, r) in enumerate(rows.iterrows()):
        user = (f"Jurisdiction: {r['place']}, {r['state'].upper()}\nSection: {label(r['header'], 160)}\n\n"
                f"Text:\n{clean(r['content'])[:3500]}")
        try:
            data = call_llm(model, host, EXTRACT_SYSTEM, user, EXTRACT_SCHEMA)
            save_annotation(r["ckey"], model, norm_annotation(data, clean(r["content"])))
        except LLMUnavailable as e:
            errs.append(str(e))
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{label(r['header'], 50)}: {e}")
        if progress:
            progress.progress((i + 1) / n, text=f"Annotated {i + 1}/{n}")
    return errs


def process_key(state, place, query, model, ckeys) -> str:
    return hashlib.sha1("|".join([state, place, query.strip().lower(), model, PROMPT_VER, *sorted(ckeys)]).encode()).hexdigest()[:20]


def compile_process(chunks: pd.DataFrame, anns: dict, state: str, place: str, query: str, model: str, host: str) -> str:
    lines, steps_rec = [], []
    for _, r in chunks.iterrows():
        a = anns.get(r["ckey"])
        for j, s in enumerate((a or {}).get("steps", []), 1):
            code = f"{r['code']}.{j}"
            steps_rec.append({"code": code, "chunk": r["code"], **s})
            lines.append(" | ".join([
                code, f"§{section_no(r['header']) or '?'}", f"actor: {s['actor'] or '?'}", f"action: {s['action']}",
                f"if: {s['condition']}" if s["condition"] else "if: -", f"due: {s['deadline'] or '-'}",
                f"fee: {s['fee'] or '-'}", f"penalty: {s['penalty'] or '-'}",
                f"refs: {', '.join(a['refs']) or '-'}"]))
    if not lines:
        raise ValueError("No extracted steps to compile. Annotate first (some sections have no actionable step).")
    user = f"Jurisdiction: {place}, {state.upper()}\nTopic: {query}\n\nSteps:\n" + "\n".join(lines)
    graph = call_llm(model, host, COMPILE_SYSTEM, user, COMPILE_SCHEMA)
    chunk_recs = [
        {"code": r["code"], "key": r["ckey"], "header": clean(r["header"]), "section": section_no(r["header"]),
         "fn": r["fn"], "topic": clean(r["topic"]), "text": clean(r["content"])[:2500],
         **{d: (None if pd.isna(r[d]) else float(r[d])) for d in DIM_LIST}}
        for _, r in chunks.iterrows()
    ]
    pkey = process_key(state, place, query, model, chunks["ckey"].tolist())
    save_process(pkey, state, place, query, model, {
        "graph": graph, "chunks": chunk_recs, "steps": steps_rec, "model": model, "prompt_ver": PROMPT_VER,
        "state": state, "place": place, "query": query})
    return pkey


# ───────────────────────────── graph normalisation ─────────────────────────────
def normalize_graph(g: dict, chunk_codes: set[str], step_codes: set[str] | None = None):
    """Sanitise an LLM graph. Node 'sources' become chunk codes; 'steps' are the cited step codes."""
    step_codes = step_codes or set()
    nodes: dict[str, dict] = {}
    order: list[str] = []
    for n in g.get("nodes", []) or []:
        nid = clean(n.get("id")).strip()
        if not nid or nid in nodes:
            continue
        t = n.get("type") if n.get("type") in ("task", "gateway", "end") else "task"
        srcs, steps = [], []
        for s in n.get("sources", []) or []:
            code = clean(s).strip()
            base = code.split(".")[0]
            if code in step_codes and code not in steps:
                steps.append(code)
            if base in chunk_codes and base not in srcs:
                srcs.append(base)
        nodes[nid] = {"id": nid, "type": t, "actor": clean(n.get("actor")).strip(),
                      "label": clean(n.get("label")).strip() or nid, "sources": srcs, "steps": steps}
        order.append(nid)
    edges, seen, dropped = [], set(), 0
    for e in g.get("edges", []) or []:
        a, b = clean(e.get("from")).strip(), clean(e.get("to")).strip()
        if a in nodes and b in nodes and a != b and (a, b) not in seen and nodes[a]["type"] != "end":
            edges.append({"src": a, "tgt": b, "label": clean(e.get("label")).strip()})
            seen.add((a, b))
        else:
            dropped += 1
    return nodes, order, edges, dropped


def node_facts(nd: dict, steps_by_code: dict) -> dict:
    """First non-empty typed/text fact across the steps a node cites."""
    out: dict[str, str] = {}
    for k in (*FACT_KEYS, "condition", "fee", "deadline", "penalty"):
        for c in nd.get("steps", []):
            v = steps_by_code.get(c, {}).get(k, "")
            if v and v != "unspecified":
                out[k] = v
                break
    return out


# ───────────────────────────── matching + summaries ─────────────────────────────
def get_matches(df: pd.DataFrame, query: str, fns: list[str], cap: int) -> pd.DataFrame:
    terms = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    if df.empty or not terms:
        return df.iloc[0:0].assign(place=[], ckey=[], code=[])
    head = df["header"].fillna("").str.lower()
    hay = head + " " + df["content"].fillna("").str.lower()
    mask = pd.Series(True, index=df.index)
    for t in terms:
        mask &= hay.str.contains(re.escape(t))
    m = df[mask & df["fn"].isin(fns)].copy()
    m["_h"] = sum(head[m.index].str.contains(re.escape(t)).astype(int) for t in terms)
    m = m.sort_values("_h", ascending=False, kind="stable").head(cap).sort_index().drop(columns="_h")
    m["place"] = m["city"].fillna(m["county"])
    m["ckey"] = [chunk_key(r.state, r.place, r.header, r.content) for r in m.itertuples()]
    m["code"] = [f"c{i + 1}" for i in range(len(m))]
    return m


def summarize(m: pd.DataFrame, anns: dict) -> dict:
    steps = [s for k in m["ckey"] for s in anns.get(k, {}).get("steps", [])]
    done = bool(anns)

    def join(vals, fmt=str):
        u = []
        for v in vals:
            if v and v not in u:
                u.append(v)
        return "; ".join(fmt(v) for v in u) if u else ("—" if done else "(not annotated)")

    return {
        "Matches": len(m),
        "Rules": int((m["fn"] == "Rules").sum()), "Process": int((m["fn"] == "Process").sum()),
        "Enforcement": int((m["fn"] == "Enforcement").sum()),
        "Mean opacity z": round(float(m["opacity"].mean()), 2) if len(m) else None,
        "Mean discretion z": round(float(m["enforcement_discretion"].mean()), 2) if len(m) else None,
        "Fee (USD)": join([s["fee_usd"] for s in steps], lambda v: f"${v}"),
        "Fee (as written)": join([s["fee"] for s in steps]),
        "Deadline (days)": join([s["deadline_days"] for s in steps]),
        "Deadline (as written)": join([s["deadline"] for s in steps]),
        "Renewal": join([s["renewal"] for s in steps if s["renewal"] != "unspecified"]),
        "Max penalty (USD)": join([s["penalty_max_usd"] for s in steps], lambda v: f"${v}"),
        "Penalty (as written)": join([s["penalty"] for s in steps]),
    }
