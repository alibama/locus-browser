"""
LOCUS explorer — local ordinances (LocalLaws/LOCUS-v1) + state/federal law (vaquill/open-us-law) -> marked-up,
verified, reviewable process models.

    pip install -r requirements.txt
    ollama pull qwen2.5:7b        # or use the Models panel in the sidebar to pull from here
    streamlit run locus_explorer.py

Hugging Face token: paste it in the sidebar (Hugging Face access); HF_TOKEN also works. Env: LOCUS_SRC, LOCUS_SLIM, LOCUS_DB,
     OLLAMA_HOST, OLLAMA_MODEL, OUL_TEMPLATE.
Docs: README.md, docs/markup-schema.md, docs/wikibase-mapping.md
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import locus_core as C
from locus_bpmn import emit_bpmn
from locus_core import DIM_LIST, DIMS, STATUSES, clean, default_host, default_model, label, sec_of, slug
from locus_hf import (OUL_REPO, apply_to_duckdb, dataset_access, find_token, fingerprint, forget_saved_token, looks_like_token,
                      mask, own_token_path, probe_duckdb, save_token, whoami)
from locus_dmn import decision_xml
from locus_llm import LLMUnavailable, model_available, model_selftest, pull_model, server_status, start_server
from locus_markup import (STAGES, Job, Progress, agreement_matrix, estimate, goal_ok, pipeline_status, process_key, run_jobs,
                          run_pipeline)
from locus_sources import OUL_CORPORA, OUL_TEMPLATE, load_law_context, load_locus_place, locus_places, locus_states

DEFAULT_SRC = os.environ.get("LOCUS_SRC", "hf://datasets/LocalLaws/LOCUS-v1/**/*.parquet")
TEMPLATE = os.environ.get("OUL_TEMPLATE", OUL_TEMPLATE)
SLIM_PATH = Path(os.environ.get("LOCUS_SLIM", "locus_slim.parquet"))
SCORES_ARE = "z-scores (standard units) from the paper's ModernBERT regressors"
HOSTED = os.environ.get("LOCUS_HOSTED", "") == "1"      # never persist tokens on a shared server


# ───────────────────────────── DuckDB plumbing ─────────────────────────────
def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def rel(path: str) -> str:
    return f"read_parquet({lit(path)}, union_by_name=true)"


@st.cache_resource
def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


@st.cache_resource
def _hf_applied() -> dict:
    return {"fp": None, "ok": True, "msg": ""}


def ensure_hf() -> tuple[str, str, dict]:
    """Make DuckDB use the current Hugging Face token (re-registers it only when it changes)."""
    token, source = find_token(st.session_state.get("hf_token", ""))
    fp, applied = fingerprint(token), _hf_applied()
    if applied["fp"] != fp:
        first = applied["fp"] is None
        ok, msg = apply_to_duckdb(_con(), token)
        applied.update(fp=fp, ok=ok, msg=msg)
        if not first:
            st.cache_data.clear()            # earlier 401/403 results must not stay cached
    return token, source, applied


def cur():
    return _con().cursor()


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    return cur().execute(sql, params or []).df()


# ───────────────────────────── cached data access ─────────────────────────────
@st.cache_data(show_spinner="Aggregating by state…")
def national_agg(source: str, sub_only: bool) -> pd.DataFrame:
    avg = ", ".join(f"avg({d}) AS {d}" for d in DIM_LIST)
    where = "WHERE is_substantive" if sub_only else ""
    return q(f"SELECT state, source_jurisdiction_type AS jtype, count(*) AS n, {avg} FROM {rel(source)} {where} GROUP BY 1, 2")


@st.cache_data(show_spinner="Counting functions and topics…")
def composition(source: str) -> pd.DataFrame:
    return q(f"""SELECT state, source_jurisdiction_type AS jtype, "function" AS fn, coalesce(topic, '(none)') AS topic, count(*) AS n
                 FROM {rel(source)} GROUP BY ALL""")


@st.cache_data
def list_states(source: str) -> list[str]:
    return locus_states(cur(), source)


@st.cache_data
def list_places(source: str, state: str) -> pd.DataFrame:
    return locus_places(cur(), source, state)


@st.cache_data(show_spinner="Reading provisions…")
def load_place(source: str, state: str, place: str) -> pd.DataFrame:
    return load_locus_place(cur(), source, state, place)


@st.cache_data(show_spinner="Searching state/federal law…")
def load_law(template: str, jur: str, corpora: tuple, terms: tuple, cap: int, pinned: tuple):
    return load_law_context(cur(), template, jur, list(corpora), list(terms), cap, list(pinned))


@st.cache_data(ttl=4, show_spinner=False)
def cached_status(host: str) -> dict:
    return server_status(host)


# ───────────────────────────── viewer ─────────────────────────────
def bpmn_viewer(xml: str, height: int = 560) -> None:
    safe = json.dumps(xml).replace("</", "<\\/")
    html = f"""
<link rel="stylesheet" href="https://unpkg.com/bpmn-js@17/dist/assets/diagram-js.css">
<link rel="stylesheet" href="https://unpkg.com/bpmn-js@17/dist/assets/bpmn-js.css">
<script src="https://unpkg.com/bpmn-js@17/dist/bpmn-navigated-viewer.production.min.js"></script>
<div id="c" style="height:{height}px;border:1px solid #d4d4d8;border-radius:6px;background:#fff"></div>
<div id="err" style="color:#b91c1c;font:13px sans-serif;padding-top:6px"></div>
<script>
  const xml = {safe};
  const viewer = new BpmnJS({{ container: '#c' }});
  viewer.importXML(xml)
    .then(() => viewer.get('canvas').zoom('fit-viewport'))
    .catch(e => {{ document.getElementById('err').textContent = 'BPMN render error: ' + e.message; }});
</script>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height + 40)
    else:
        components.html(html, height=height + 40)


# ───────────────────────────── model server panel (sidebar) ─────────────────────────────
def models_panel() -> dict:
    st.header("Models")
    host = st.text_input("Ollama host", default_host())
    status = cached_status(host)
    if not status["up"]:
        st.error(f"Ollama is not reachable at {host}: {status['error']}")
        c1, c2 = st.columns(2)
        if c1.button("Start Ollama here", help="Runs `ollama serve` on this machine (needs the ollama binary on PATH)"):
            ok, msg = start_server(host)
            (st.success if ok else st.error)(msg)
            cached_status.clear()
            if ok:
                st.rerun()
        c2.link_button("Install Ollama", "https://ollama.com/download")
        st.caption("Stored results stay viewable without a model; running new stages needs one.")
    else:
        st.success(f"Ollama is up · {len(status['models'])} model(s) installed")
        with st.expander("Installed models"):
            st.dataframe(pd.DataFrame([{"model": m["name"], "GB": round(m["size"] / 1e9, 1)} for m in status["models"]]))

    models_text = st.text_area("Models to compare (one per line, up to 4)", default_model(), key="models_text", height=90,
                               help="Each model gets its own stored results; nothing is overwritten. The first line is the default.")
    models = list(dict.fromkeys(m.strip() for m in models_text.splitlines() if m.strip()))[:4]
    verifier = st.text_input("Verifier (optional)", "", key="verifier",
                             help="Checks every cited rule against its source. A different model than the extractor catches more.").strip()
    wanted = [m for m in dict.fromkeys([*models, verifier]) if m]
    for i, m in enumerate(wanted):
        have = model_available(status, m) if status["up"] else None
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"`{m}` — " + ("✅ installed" if have else "⬇ not installed" if have is False else "❔ server down"))
        if status["up"] and not have and c2.button("Pull", key=f"pull{i}"):
            bar, msg = st.progress(0.0), st.empty()
            try:
                for d in pull_model(host, m):
                    tot = d.get("total") or 0
                    bar.progress(min(1.0, (d.get("completed") or 0) / tot) if tot else 0.0)
                    msg.caption(d.get("status", ""))
                cached_status.clear()
                st.rerun()
            except LLMUnavailable as e:
                st.error(str(e))
        elif status["up"] and have and c2.button("Self-test", key=f"test{i}", help="Tiny schema-following check"):
            r = model_selftest(host, m)
            (st.success if r["ok"] else st.error)(f"{m}: {'OK' if r['ok'] else r['error']} ({r['seconds']}s)")
    parallel = st.checkbox("Run models at the same time", len(models) <= 2, key="parallel",
                           help="One worker thread per model. This only saves time if every model fits in memory together AND "
                                "the hardware has spare capacity; on one GPU it usually just makes each model slower. Check "
                                "`ollama ps` for the CPU/GPU split.")
    workers = int(st.number_input("Requests in flight per model", 1, 4, 1,
                                  help="Raise only if Ollama is started with OLLAMA_NUM_PARALLEL > 1."))
    timeout = int(st.number_input("Per-call timeout (seconds)", 120, 7200, 1800, step=120,
                                  help="A slow machine can need many minutes for one call on a larger model."))
    return {"host": host, "models": models, "verifier": verifier, "parallel": parallel, "workers": workers, "timeout": timeout,
            "status": status}


# ───────────────────────────── running jobs with live progress ─────────────────────────────
def run_with_progress(jobs: list[Job], parallel: bool) -> dict:
    prog = Progress()
    res: dict = {}
    t = threading.Thread(target=lambda: res.update(run_jobs(jobs, prog, parallel)), daemon=True)
    t.start()
    bar, box = st.progress(0.0, text="Starting…"), st.empty()
    while t.is_alive():
        snap = prog.snapshot()
        frac, lines = 0.0, []
        for j in jobs:
            s = snap.get(j.pid, {})
            stage, done, total = s.get("stage", "queued"), s.get("done", 0), s.get("total", 0)
            idx = STAGES.index(stage) if stage in STAGES else (len(STAGES) if stage == "done" else 0)
            frac += (idx + (done / total if total else 0)) / len(STAGES)
            lines.append(f"- **{j.pid}** — {stage}" + (f" {done}/{total}" if total else ""))
        bar.progress(min(1.0, frac / max(1, len(jobs))), text="Working…")
        box.markdown("\n".join(lines))
        time.sleep(0.35)
    t.join()
    bar.empty()
    box.empty()
    st.session_state["run_log"] = {k: v["errors"] for k, v in res.items() if v["errors"]}
    return res


def show_run_log() -> None:
    log = st.session_state.get("run_log")
    if log:
        with st.expander("Messages from the last run", expanded=True):
            for k, errs in log.items():
                for e in errs:
                    st.warning(f"{k}: {e}")


# ───────────────────────────── candidates + cells ─────────────────────────────
def gather(src, state, place, query, cfg) -> tuple[pd.DataFrame, list[str]]:
    local = load_place(src, state, place) if place else pd.DataFrame()
    frames = {"local": local}
    problems: list[str] = []
    terms = tuple(t for t in query.lower().split() if t)
    sterms = tuple(t for t in (cfg.get("state_query") or query).lower().split() if t)
    pinned = tuple(p.strip() for p in cfg["pinned"].split(",") if p.strip())
    if cfg["state_law"] and (sterms or pinned):
        frames["state"], pr = load_law(cfg["template"], state, tuple(cfg["corpora"]), sterms, cfg["cap_state"], pinned)
        problems += pr
    if cfg["federal_law"] and terms:
        frames["federal"], pr = load_law(cfg["template"], cfg["fed_code"], tuple(cfg["corpora"]), terms, cfg["cap_fed"], ())
        problems += pr
    caps = {"local": cfg["cap_local"], "state": cfg["cap_state"], "federal": cfg["cap_fed"]}
    fns = ["Rules", "Process", "Enforcement"] + (["Context"] if cfg["ctx"] else [])
    return C.build_candidates(frames, query, caps, fns, strict=cfg.get("strict", True)), problems


def make_job(m, state, place, query, model, mcfg, goal: str = "") -> Job:
    return Job(state=state, place=place, query=query, chunks=m, model=model, host=mcfg["host"],
               verifier=mcfg["verifier"], workers=mcfg["workers"], goal=goal, timeout=mcfg["timeout"])


def stage_line(stt: dict) -> str:
    tick = lambda ok: "✓" if ok else "·"
    return (f"triage {stt['triage']}/{stt['chunks']} · relevance {stt['relevance']}/{stt['chunks']} "
            f"({stt['relevant']} relevant) · extract {stt['extract']}/{stt['relevant']} · regime {tick(stt['regime'])} · "
            f"compile {tick(stt['process'])} · verified {stt['verified']}")


def section_table(job: Job, stt: dict) -> pd.DataFrame:
    rows = []
    for _, r in job.chunks.iterrows():
        k = r["ckey"]
        t, rl, ex = stt["triage_data"].get(k, {}), stt["relevance_data"].get(k, {}), stt["extract_data"].get(k, {})
        rows.append({"code": r["code"], "level": r["level"], "§": sec_of(r), "title": label(r["header"], 50),
                     "role": t.get("role", ""), "audience": t.get("audience", ""), "relevant": rl.get("verdict", ""),
                     "why": rl.get("reason", ""), "subjects": ", ".join(t.get("subjects", [])),
                     "extracted": str(len(ex.get("rules", [])) + len(ex.get("rows", []))) if ex else ""})
    return pd.DataFrame(rows)


def cell(job: Job, kp: str, color_dim: str | None, height: int = 520) -> None:
    stt = pipeline_status(job)
    st.caption(stage_line(stt))
    qi = C.query_get(job.query_key, job.model)
    if qi:
        (st.caption if qi.get("by") != "fallback" else st.warning)(
            f"Goal used ({qi.get('by', 'llm')}): {qi['goal']}" + ("  — the model's guess was unusable; type a goal above." if qi.get("by") == "fallback" else ""))
    if not stt["process"]:
        eta = estimate(job)
        st.caption(f"Estimated time left: about {eta['minutes']} min for ~{eta['calls']} model calls (based on {eta['basis']}).")
    c1, c2, c3 = st.columns([2, 2, 2])
    if c1.button("Run pipeline (resumes)", key=kp + "run"):
        run_with_progress([job], False)
        st.rerun()
    if c2.button("Re-verify", key=kp + "ver", disabled=not stt["process"], help="Run the verifier again on every cited rule"):
        run_pipeline(job, Progress(), ["verify"], force_verify=True)
        st.rerun()
    if c3.button("Re-compile", key=kp + "rec", disabled=not stt["process"],
                 help="Rebuild the graph from the stored markup (keeps your review status; clears verdicts, then re-verifies)"):
        run_pipeline(job, Progress(), ["compile", "verify"], force_compile=True)
        st.rerun()
    with st.expander("Sections: what the pipeline decided about each", expanded=not stt["process"]):
        st.dataframe(section_table(job, stt))
        for _, r in job.chunks.iterrows():
            t = stt["triage_data"].get(r["ckey"])
            if t and t.get("summary"):
                st.markdown(f"**{r['code']} §{sec_of(r) or '?'}** · {t['role']} — {t['summary']}")
            with st.popover(f"{r['code']} text"):
                st.markdown(clean(r["content"])[:4000])
    rec = C.get_process(stt["pkey"])
    if not rec:
        return
    reg = rec["regime"]
    with st.expander("Regime outline (phases, excluded, missing)"):
        st.write({p["phase"]: p["codes"] for p in reg["phases"]})
        for x in reg.get("excluded", []):
            st.markdown(f"- excluded **{x['code']}** — {x['reason']}")
        for x in reg.get("missing", []):
            st.markdown(f"- **not found in the sources:** {x['item']} — {x['why']}")
    xml, meta, warns = emit_bpmn(rec, job.state, job.place, job.query, color_dim)
    if xml is None:
        st.warning(" ".join(warns))
        return
    bpmn_viewer(xml, height)
    st.caption("Lanes = who acts. Grey = no source cited · red = verifier says the text does not state it · amber outline = partly "
               "supported. " + (f"Fill = mean {color_dim.replace('_', ' ')} of the cited provisions." if color_dim else ""))
    for w in warns:
        st.warning(w)
    r1, r2, r3 = st.columns([1, 1, 2])
    status = r1.selectbox("Review status", STATUSES, index=STATUSES.index(rec["status"]), key=kp + "st")
    reviewer = r2.text_input("Reviewer", rec.get("reviewer", ""), key=kp + "rv")
    note = r3.text_input("What was checked", rec["note"], key=kp + "nt")
    if (status, reviewer, note) != (rec["status"], rec.get("reviewer", ""), rec["note"]) and st.button("Save review", key=kp + "sv"):
        C.set_review(stt["pkey"], status, note, reviewer)
        st.rerun()
    fname = f"{job.state}-{slug(job.place)}-{slug(job.query)}-{slug(job.model)}"
    d = st.columns(4)
    d[0].download_button("BPMN", xml, f"{fname}.bpmn", "application/xml", key=kp + "dx")
    d[1].download_button("Sidecar JSON", json.dumps({"pkey": stt["pkey"], "status": rec["status"], "elements": meta}, indent=2),
                         f"{fname}.json", "application/json", key=kp + "dj")
    d[2].download_button("Debug bundle", json.dumps(C.export_debug(stt["pkey"]), indent=2, default=str), f"{fname}-debug.json",
                         "application/json", key=kp + "dd",
                         help="Every stage output + every prompt/response. Send this back when something looks wrong.")
    for dec in rec["decisions"]:
        d[3].download_button(f"DMN {dec['id']}", decision_xml(dec, job.place, job.state), f"{fname}-{dec['id']}.dmn",
                             "application/xml", key=kp + "dm" + dec["id"], help=f"Decision table: {dec['name']}")
    with st.expander("Elements and sources"):
        st.dataframe(pd.DataFrame(meta).astype(str))
    export_ui(rec, kp)


def export_ui(rec: dict, kp: str) -> None:
    from lexipedia_export import export_files, exportable, load_props, resolve_subject

    with st.expander("Wikibase export (dry run — writes nothing)"):
        if rec["status"] != "reviewed-ok" or not rec.get("reviewer") or not rec.get("note"):
            st.info("Set status **reviewed-ok** with a reviewer and a note, then save. Draft LLM output is never exported.")
            return
        reg = C.subjects(include_proposed=False)
        try:
            default = resolve_subject(rec)[0]
        except ValueError:
            default = next(iter(reg))
        subject = st.selectbox("Regulated subject (controlled vocabulary)", list(reg),
                               index=list(reg).index(default) if default in reg else 0,
                               format_func=lambda s: f"{reg[s]['label']} ({s})", key=kp + "subj")
        partial = st.checkbox("Also export partially-supported steps", False, key=kp + "part")
        raw = st.text_area("qid_map.json — local key → QID (classes, units, modalities, jurisdictions; later also pass-1 results)",
                           "{}", key=kp + "qm", height=90)
        try:
            qm = json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            st.error(f"qid_map is not valid JSON: {e}")
            return
        bundle, p1, p2, report = export_files(rec, subject, qm, load_props(), include_partial=partial)
        keep, held = exportable(rec, partial)
        st.caption(f"{len(bundle['items'])} items · {len(keep)} step(s) exported · pass 1: {p1.count(chr(10))} lines · "
                   f"pass 2: {p2.count(chr(10))} lines")
        for h in held:
            st.warning(f"Held back — {h}")
        for k in report["pass1"]["unresolved"]:
            st.warning(f"Needs a QID in qid_map: `{k}`")
        for w in report["pass1"]["warnings"]:
            st.warning(w)
        c = st.columns(3)
        c[0].download_button("bundle.json", json.dumps(bundle, indent=2), f"{kp}bundle.json", "application/json", key=kp + "eb")
        c[1].download_button("pass1.qs", p1, f"{kp}pass1.qs", "text/plain", key=kp + "e1")
        c[2].download_button("pass2.qs", p2, f"{kp}pass2.qs", "text/plain", key=kp + "e2")


def grid_cols(n: int, per_row: int) -> list:
    out = []
    for start in range(0, n, per_row):
        out += st.columns(min(per_row, n - start))
    return out


def _use_token() -> None:
    t = st.session_state.get("hf_token_input", "").strip()
    if t:
        st.session_state["hf_token"] = t
        if st.session_state.get("hf_remember") and not HOSTED:
            save_token(t)
    st.session_state["hf_token_input"] = ""


def _forget_token() -> None:
    st.session_state["hf_token"] = ""
    forget_saved_token()
    st.cache_data.clear()


def hf_panel(template: str) -> None:
    token, source, applied = ensure_hf()
    with st.expander("Hugging Face access", expanded=not token):
        if token:
            st.success(f"Using a token from **{source}** ({mask(token)})")
        else:
            st.warning("No token yet. open-us-law is gated, so state and federal law will not load without one.")
        st.text_input("Paste your Hugging Face token", type="password", key="hf_token_input", placeholder="hf_…",
                      help="Create a token with Read access at huggingface.co/settings/tokens. It stays in this session "
                           "unless you tick 'Remember'.")
        st.checkbox("Remember on this computer", key="hf_remember", disabled=HOSTED,
                    help="Saves the token to " + str(own_token_path()) + (" (disabled on a shared server)" if HOSTED else ""))
        c1, c2, c3 = st.columns(3)
        c1.button("Use token", on_click=_use_token, key="hf_use")
        check = c2.button("Check access", key="hf_check")
        c3.button("Forget", on_click=_forget_token, key="hf_forget", help="Clears the session token and the one this app saved. "
                  "A token set via an environment variable or `huggingface-cli login` is left alone.")
        typed = st.session_state.get("hf_token_input", "")
        if typed and not looks_like_token(typed):
            st.caption("That does not look like a Hugging Face token (they start with `hf_`).")
        l1, l2 = st.columns(2)
        l1.link_button("Create a token", "https://huggingface.co/settings/tokens")
        l2.link_button("Accept dataset terms", f"https://huggingface.co/datasets/{OUL_REPO}")
        st.caption("No environment variable is needed. On Windows the app reads the token you paste here.")
        if not applied["ok"]:
            st.error(applied["msg"])
        if check:
            who = whoami(token)
            st.markdown(("✅ " if who["ok"] else "❌ ") + (f"Signed in as **{who['name']}**" if who["ok"] else who["error"]))
            acc = dataset_access(token)
            st.markdown(("✅ " if acc["state"] == "ok" else "❌ ") + acc["message"])
            pr = probe_duckdb(cur(), template)
            st.markdown(("✅ " if pr["ok"] else "❌ ") + (f"DuckDB read `{pr['path'].rsplit('/', 1)[-1]}` ({pr['rows']:,} rows)"
                                                       if pr["ok"] else f"DuckDB could not read `{pr['path']}`: {pr['error']}"))


def law_hint(problem: str) -> str:
    low = problem.lower()
    if any(x in low for x in ("403", "401", "forbidden", "unauthorized", "gated", "authentication")):
        return problem + "  → open **Hugging Face access** in the sidebar and use *Check access*."
    return problem


def sources_panel() -> dict:
    ensure_hf()
    st.header("Data")
    src = st.text_input("LOCUS parquet (local ordinances)", DEFAULT_SRC, help="Local path, glob, or hf:// glob")
    agg_src = str(SLIM_PATH) if SLIM_PATH.exists() else src
    st.caption(f"National views read: `{agg_src}`")
    if st.button("Build slim cache (no text, fast national views)"):
        cols = ", ".join(["state", "city", "county", "source_jurisdiction_type", '"function"', "topic", "is_substantive", *DIM_LIST])
        with st.spinner("Streaming columns from source — one-time, can take a while…"):
            cur().execute(f"COPY (SELECT {cols} FROM {rel(src)}) TO {lit(str(SLIM_PATH))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        st.cache_data.clear()
        st.rerun()
    sub_only = st.checkbox("National view: substantive chunks only", value=True)
    with st.expander("State / federal law (open-us-law)"):
        st.caption("Files are named `us_{jurisdiction}_{corpus}.parquet`. Gated: see *Hugging Face access* below.")
        state_law = st.checkbox("Include state law for the selected state", True)
        federal_law = st.checkbox("Include federal law", False)
        template = st.text_input("File template", TEMPLATE)
        corpora = st.multiselect("Corpora", OUL_CORPORA, default=["statutes"])
        fed_code = st.text_input("Federal jurisdiction code in file names", "us", help="Check the dataset's Files tab: us, federal, …")
        state_query = st.text_input("State-law search words (blank = same as topic)", "",
                                    help="State statutes use different words than local codes, e.g. 'local license tax gross receipts'.")
        pinned = st.text_input("Always include citations containing", "", help="Comma-separated, e.g. 58.1-37 for Virginia's local-tax chapter")
    hf_panel(template)
    c1, c2, c3 = st.columns(3)
    strict = st.checkbox("Local sections must contain the topic phrase (or have it in the title)", True,
                         help="Stops sections that merely mention both words somewhere (e.g. 'Fixing payday') from being sent to the model.")
    cap_local = int(c1.number_input("Local max", 1, 60, 15))
    cap_state = int(c2.number_input("State max", 0, 30, 6))
    cap_fed = int(c3.number_input("Fed max", 0, 20, 0))
    return {"src": src, "agg_src": agg_src, "sub_only": sub_only, "state_law": state_law, "federal_law": federal_law,
            "template": template, "corpora": corpora or ["statutes"], "fed_code": fed_code, "pinned": pinned, "state_query": state_query, "strict": strict,
            "cap_local": cap_local, "cap_state": cap_state, "cap_fed": cap_fed, "ctx": False}


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    st.set_page_config(page_title="LOCUS explorer", layout="wide")
    st.title("LOCUS explorer")
    st.caption("Local ordinances (LOCUS-v1) + state/federal law (open-us-law) → marked-up, verified, reviewable process models · "
               "dimension scores are " + SCORES_ARE)

    with st.sidebar:
        cfg = sources_panel()
        mcfg = models_panel()
        color_choice = st.selectbox("Colour process boxes by", ["(none)", *DIM_LIST])
        color_dim = None if color_choice == "(none)" else color_choice

    src, agg_src = cfg["src"], cfg["agg_src"]
    try:
        states = list_states(agg_src)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read `{agg_src}`.\n\n{e}")
        st.info("Check the path/glob, paste a Hugging Face token in the sidebar (Hugging Face access) if the dataset is gated, or point LOCUS_SRC at a local parquet.")
        st.stop()

    t_nat, t_jur, t_prov, t_proc, t_cmp, t_lib = st.tabs(
        ["National view", "Jurisdiction", "Provisions", "Process builder", "Compare", "Library"])

    with t_nat:
        agg = national_agg(agg_src, cfg["sub_only"])
        g = agg.copy()
        for d in DIM_LIST:
            g[d] = g[d] * g["n"]
        sm = g.groupby("state")[[*DIM_LIST, "n"]].sum()
        for d in DIM_LIST:
            sm[d] = sm[d] / sm["n"]
        sm = sm.reset_index()
        dim = st.selectbox("Dimension", DIM_LIST, format_func=lambda d: DIMS[d])
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.choropleth(sm, locations=sm["state"].str.upper(), locationmode="USA-states", color=dim, scope="usa",
                                color_continuous_scale="RdBu_r", color_continuous_midpoint=0, hover_data={"n": True}, labels={dim: "mean z"})
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)
        with c2:
            by_type = agg.assign(w=agg[dim] * agg["n"]).groupby("jtype")[["w", "n"]].sum()
            by_type["mean z"] = by_type["w"] / by_type["n"]
            st.plotly_chart(px.bar(by_type.reset_index(), x="jtype", y="mean z", title="Cities vs counties"))
            st.plotly_chart(px.bar(sm.sort_values(dim, ascending=False).head(10), x=dim, y="state", orientation="h", title="Top 10 states"))
        st.subheader("What each kind of jurisdiction regulates")
        comp = composition(agg_src)
        comp = comp[comp["fn"].isin(["Rules", "Enforcement"])] if cfg["sub_only"] else comp
        share = comp.groupby(["jtype", "topic"])["n"].sum().reset_index()
        share["share"] = share["n"] / share.groupby("jtype")["n"].transform("sum")
        st.plotly_chart(px.bar(share, x="jtype", y="share", color="topic", barmode="stack"))

    with st.sidebar:
        st.header("Jurisdiction")
        state = st.selectbox("State", states, index=states.index("va") if "va" in states else 0)
        plist = list_places(agg_src, state)["place"].tolist()
        place = st.selectbox("City / county", plist, index=plist.index("charlottesville") if "charlottesville" in plist else 0)
    df = load_place(src, state, place) if place else pd.DataFrame()

    with t_jur:
        if df.empty:
            st.warning("No rows for that jurisdiction.")
        else:
            m4 = st.columns(4)
            m4[0].metric("Chunks", f"{len(df):,}")
            m4[1].metric("Substantive", f"{df['is_substantive'].astype(float).mean():.0%}")
            m4[2].metric("Type", ", ".join(sorted(df["jtype"].dropna().unique())))
            m4[3].metric("Mean opacity (z)", f"{df['opacity'].mean():+.2f}")
            a, b = st.columns(2)
            a.plotly_chart(px.histogram(df, x="fn", color="fn", title="Function"))
            b.plotly_chart(px.histogram(df[df["is_substantive"].astype(bool)], x="topic", color="topic", title="Topic (substantive only)"))
            dsel = st.selectbox("Dimension by topic", DIM_LIST, format_func=lambda d: DIMS[d], key="jd")
            st.plotly_chart(px.box(df, x="topic", y=dsel, color="fn", points="outliers"))
            st.plotly_chart(px.scatter(df, x="opacity", y="paternalism", color="topic", hover_name="header", opacity=0.6,
                                       title="Opacity vs paternalism"))
            st.subheader("Hardest to read (highest opacity)")
            st.dataframe(df.sort_values("opacity", ascending=False)[["header", "fn", "topic", "opacity", "enforcement_discretion"]].head(15))

    with t_prov:
        if not df.empty:
            f1, f2, f3, f4 = st.columns([3, 2, 2, 1])
            text_q = f1.text_input("Search header / text")
            fns = f2.multiselect("Function", sorted(df["fn"].dropna().unique()))
            tps = f3.multiselect("Topic", sorted(df["topic"].dropna().unique()))
            hi = f4.number_input("Min opacity z", value=-5.0, step=0.5)
            v = df
            if text_q:
                v = v[v["header"].str.contains(text_q, case=False, na=False) | v["content"].str.contains(text_q, case=False, na=False)]
            if fns:
                v = v[v["fn"].isin(fns)]
            if tps:
                v = v[v["topic"].isin(tps)]
            v = v[v["opacity"].fillna(-99) >= hi]
            st.caption(f"{len(v):,} of {len(df):,} chunks")
            ev = st.dataframe(v[["header", "fn", "topic", *DIM_LIST]], on_select="rerun", selection_mode="single-row", key="prov_tbl")
            if ev.selection.rows:
                r = v.iloc[ev.selection.rows[0]]
                st.markdown(f"**{label(r['header'], 200)}**")
                st.markdown(r["content"])

    models = mcfg["models"] or [default_model()]

    # ── Process builder (one place, 1-2 models) ──
    with t_proc:
        st.markdown(
            "**Markup → relevance → extract → outline → compile → verify.** Each pass is stored, so re-runs are free and "
            "two models never overwrite each other. Nothing here is a legal reading; it is a draft for a lawyer to check.")
        show_run_log()
        c1, c2 = st.columns([4, 1])
        pq = c1.text_input("Topic (all words must appear; stems work)", "business licen", key="pb_q")
        cfg["ctx"] = c2.checkbox("Include Context", False, key="pb_ctx")
        pgoal = st.text_input("What process are you mapping? (one sentence — optional; the model guesses if blank)", "", key="pb_goal",
                              placeholder="e.g. Start a new business in Charlottesville: what must the owner do, pay and file?")
        if pgoal.strip() and not goal_ok(pgoal):
            st.caption("A full sentence works best (at least 4 words).")
        m, problems = gather(src, state, place, pq, cfg)
        for p in problems:
            st.warning("open-us-law: " + law_hint(p))
        if m.empty:
            st.info("No matching sections. Try fewer or shorter words, or untick the topic-phrase filter.")
        else:
            st.caption(f"{len(m)} candidate sections: " + " / ".join(f"{int((m['level'] == lv).sum())} {lv}" for lv in ("local", "state", "federal")))
            jobs = [make_job(m, state, place, pq, mod, mcfg, pgoal) for mod in models]
            if len(jobs) > 1 and st.button("Run all models now", key="pb_all"):
                run_with_progress(jobs, mcfg["parallel"])
                st.rerun()
            for i, (col, job) in enumerate(zip(grid_cols(len(jobs), 2), jobs)):
                with col:
                    st.markdown(f"#### {job.model}")
                    cell(job, f"pb{i}_", color_dim)
            if len(jobs) > 1:
                rows = agreement_matrix({j.model: C.get_process(process_key(j)) for j in jobs})
                if rows:
                    st.subheader("Do the models agree?")
                    st.dataframe(pd.DataFrame(rows).astype(str))

    # ── Compare: places × models ──
    with t_cmp:
        st.markdown("Same topic, several places, and (optionally) two models at once.")
        show_run_log()
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        cq = c1.text_input("Topic (all words must appear; stems work)", "dog licen", key="cmp_q")
        cfg["ctx"] = c2.checkbox("Include Context", False, key="cmp_ctx")
        n = int(c3.number_input("Places to compare", 2, 8, 2, key="cmp_n"))
        per_row = int(c4.number_input("Columns per row", 1, 4, 3, key="cmp_cols"))
        cgoal = st.text_input("What process are you comparing? (optional sentence)", "", key="cmp_goal")
        picks: list[tuple[str, str]] = []
        for i, col in enumerate(grid_cols(n, per_row)):
            with col:
                dstate = state if state in states else states[0]
                s = st.selectbox("State", states, index=states.index(dstate), key=f"cs{i}")
                pl = list_places(agg_src, s)["place"].tolist()
                prefer = ["charlottesville", "albemarle"][i] if i < 2 else None
                idx = pl.index(prefer) if prefer in pl else min(i, max(len(pl) - 1, 0))
                picks.append((s, st.selectbox("City / county", pl, index=idx, key=f"cp{i}")))

        cand = []
        for s, p in picks:
            mm, pr = gather(src, s, p, cq, cfg)
            cand.append((s, p, mm))
            for x in pr:
                st.warning(f"{p}: open-us-law: " + law_hint(x))
        all_jobs = [make_job(mm, s, p, cq, mod, mcfg, cgoal) for s, p, mm in cand if len(mm) for mod in models]
        if all_jobs and st.button(f"Run everything ({len(cand)} places × {len(models)} model(s))", key="cmp_all"):
            run_with_progress(all_jobs, mcfg["parallel"])
            st.rerun()

        cols_tbl = {}
        for s, p, mm in cand:
            for mod in models:
                ex = C.markup_get(mm["ckey"].tolist(), "extract", mod) if len(mm) else {}
                cols_tbl[f"{p.title()}, {s.upper()} · {mod}"] = C.summarize(mm, ex, mod) if len(mm) else {"Sections": 0}
        st.subheader("At a glance")
        st.dataframe(pd.DataFrame(cols_tbl).astype(str))

        if len(models) > 1:
            rows = []
            for s_, p_, mm in cand:
                if len(mm):
                    for r_ in agreement_matrix({mod: C.get_process(process_key(make_job(mm, s_, p_, cq, mod, mcfg, cgoal))) for mod in models}):
                        rows.append({"place": f"{p_.title()}, {s_.upper()}", **{k: str(v) for k, v in r_.items()}})
            if rows:
                st.subheader("Do the models agree?")
                st.dataframe(pd.DataFrame(rows))

        st.subheader("Process models")
        for pi, (s, p, mm) in enumerate(cand):
            st.markdown(f"### {p.title()}, {s.upper()}")
            if mm.empty:
                st.info("No matching sections here.")
                continue
            for col, (mi, mod) in zip(grid_cols(len(models), 2), enumerate(models)):
                with col:
                    st.markdown(f"**{mod}**")
                    cell(make_job(mm, s, p, cq, mod, mcfg, cgoal), f"cmp{pi}_{mi}_", color_dim, height=460)

    # ── Library ──
    with t_lib:
        st.subheader("Subject registry (controlled vocabulary)")
        st.caption("Triage tags sections with these slugs. New ones arrive as *proposed*; approve them here. The QID links a subject "
                   "to its Wikibase item and is what makes processes comparable across places.")
        reg = C.subjects()
        st.dataframe(pd.DataFrame([{"slug": k, **v} for k, v in reg.items()]))
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        sl = c1.selectbox("Subject", list(reg), key="lib_s")
        qid = c2.text_input("Wikibase QID", reg[sl]["qid"], key="lib_q")
        if c3.button("Save QID"):
            C.subject_set(sl, qid=qid)
            st.rerun()
        if c4.button("Approve", disabled=reg[sl]["status"] != "proposed"):
            C.subject_set(sl, status="approved")
            st.rerun()
        new = st.text_input("Add a subject (label)", key="lib_new")
        if st.button("Add") and new:
            C.subject_add(slug(new), new, "approved")
            st.rerun()
        st.subheader("Stored processes")
        procs = C.list_processes()
        st.dataframe(pd.DataFrame(procs))
        if procs:
            pk = st.selectbox("Process", [p["pkey"] for p in procs], key="lib_p")
            st.download_button("Debug bundle (all stages, prompts, responses)", json.dumps(C.export_debug(pk), indent=2, default=str),
                               f"{pk}-debug.json", "application/json")


if __name__ == "__main__":
    main()
