"""
LOCUS explorer — Streamlit app for LocalLaws/LOCUS-v1 (Hugging Face)

    pip install -r requirements.txt
    ollama pull qwen2.5:7b          # or any model; set in the sidebar
    streamlit run locus_explorer.py

Env: HF_TOKEN (gated datasets), LOCUS_SRC (parquet path/glob/hf:// glob), LOCUS_SLIM (text-free cache),
     LOCUS_DB (SQLite for annotations + processes), OLLAMA_HOST, OLLAMA_MODEL.

Tabs: National view / Jurisdiction / Provisions / Process builder / Compare (2-8 places).
See README.md and docs/wikibase-mapping.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from locus_bpmn import emit_bpmn
from locus_core import (DIM_LIST, DIMS, STATUSES, LLMUnavailable, annotate, clean, compile_process, default_host,
                        default_model, get_annotations, get_matches, get_process, label, ollama_models,
                        process_key, section_no, set_review, slug, summarize)

DEFAULT_SRC = os.environ.get("LOCUS_SRC", "hf://datasets/LocalLaws/LOCUS-v1/**/*.parquet")
SLIM_PATH = Path(os.environ.get("LOCUS_SLIM", "locus_slim.parquet"))
SCORES_ARE = "z-scores (standard units) from the paper's ModernBERT regressors"


# ───────────────────────────── DuckDB plumbing ─────────────────────────────
def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def rel(path: str) -> str:
    return f"read_parquet({lit(path)}, union_by_name=true)"


@st.cache_resource
def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        pass
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        try:
            con.execute(f"CREATE OR REPLACE SECRET hf_tok (TYPE HUGGINGFACE, TOKEN {lit(tok)})")
        except Exception:
            pass
    return con


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    return _con().cursor().execute(sql, params or []).df()


def run(sql: str) -> None:
    _con().cursor().execute(sql)


# ───────────────────────────── cached queries ─────────────────────────────
@st.cache_data(show_spinner="Aggregating by state…")
def national_agg(source: str, sub_only: bool) -> pd.DataFrame:
    avg = ", ".join(f"avg({d}) AS {d}" for d in DIM_LIST)
    where = "WHERE is_substantive" if sub_only else ""
    return q(
        f"""SELECT state, source_jurisdiction_type AS jtype, count(*) AS n, {avg}
            FROM {rel(source)} {where} GROUP BY 1, 2"""
    )


@st.cache_data(show_spinner="Counting functions and topics…")
def composition(source: str) -> pd.DataFrame:
    return q(
        f"""SELECT state, source_jurisdiction_type AS jtype, "function" AS fn,
                   coalesce(topic, '(none)') AS topic, count(*) AS n
            FROM {rel(source)} GROUP BY ALL"""
    )


@st.cache_data
def list_states(source: str) -> list[str]:
    df = q(f"SELECT DISTINCT state FROM {rel(source)} WHERE state IS NOT NULL ORDER BY 1")
    return df["state"].tolist()


@st.cache_data
def list_places(source: str, state: str) -> pd.DataFrame:
    return q(
        f"""SELECT DISTINCT coalesce(city, county) AS place, source_jurisdiction_type AS jtype
            FROM {rel(source)} WHERE state = ? AND coalesce(city, county) IS NOT NULL
            ORDER BY 1""",
        [state],
    )


@st.cache_data(show_spinner="Reading provisions…")
def load_place(source: str, state: str, place: str) -> pd.DataFrame:
    dims = ", ".join(DIM_LIST)
    df = q(
        f"""SELECT header, content, is_substantive, "function" AS fn, topic,
                   source_jurisdiction_type AS jtype, state, city, county, {dims}
            FROM {rel(source)} WHERE state = ? AND coalesce(city, county) = ?""",
        [state, place],
    )
    return df.reset_index(drop=True)


# ───────────────────────────── viewer + panels ─────────────────────────────
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
    if hasattr(st, "iframe"):  # newer Streamlit; components.html is being retired
        st.iframe(html, height=height + 40)
    else:
        components.html(html, height=height + 40)


def wikibase_export_ui(rec: dict, kp: str) -> None:
    from lexipedia_export import export_files, load_props

    with st.expander("Wikibase export (dry run — writes nothing)"):
        if rec["status"] != "reviewed-ok":
            st.info("Set the review status to **reviewed-ok** and save. Draft LLM output is not exported.")
            return
        props = load_props()
        subject = st.text_input("Subject label (shared across places — this is what makes them comparable)",
                                rec["graph"].get("title") or rec["query"], key=kp + "subj")
        raw = st.text_area("qid_map.json — local key → QID for items that already exist (classes, units, "
                           "jurisdictions, and pass-1 results when running pass 2)", "{}", key=kp + "qm", height=90)
        try:
            qm = json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            st.error(f"qid_map is not valid JSON: {e}")
            return
        bundle, p1, p2, report = export_files(rec, subject, qm, props)
        st.caption(f"{len(bundle['items'])} items in bundle · pass 1: {p1.count(chr(10))} lines · pass 2: {p2.count(chr(10))} lines")
        for k in report["pass1"]["unresolved"]:
            st.warning(f"Needs a QID in qid_map: `{k}`")
        for w in report["pass1"]["warnings"]:
            st.warning(w)
        c = st.columns(3)
        c[0].download_button("bundle.json", json.dumps(bundle, indent=2), f"{kp}bundle.json", "application/json", key=kp + "eb")
        c[1].download_button("pass1.qs", p1, f"{kp}pass1.qs", "text/plain", key=kp + "e1")
        c[2].download_button("pass2.qs", p2, f"{kp}pass2.qs", "text/plain", key=kp + "e2")
        with st.popover("Preview pass 1"):
            st.code(p1 or "(empty — resolve the QIDs above first)", language="text")


def process_panel(m: pd.DataFrame, state: str, place: str, query: str, kp: str,
                  model: str, host: str, color_dim: str | None, height: int = 520) -> None:
    if m.empty:
        st.info("No matching provisions. Try fewer or shorter words (stems work: 'licen').")
        return
    anns = get_annotations(m["ckey"].tolist(), model)
    missing = m[~m["ckey"].isin(anns)]
    st.caption(f"{len(m)} provisions · {len(anns)} annotated by `{model}`")

    b1, b2 = st.columns(2)
    if b1.button(f"1 · Annotate {len(missing)} provision(s)", key=kp + "ann", disabled=missing.empty):
        bar = st.progress(0.0, text="Calling Ollama…")
        errs = annotate(missing, model, host, bar)
        bar.empty()
        for e in errs:
            st.error(e)
        if not errs:
            st.rerun()
    pkey = process_key(state, place, query, model, m["ckey"].tolist())
    rec = get_process(pkey)
    if b2.button("2 · Compile process" + (" (redo)" if rec else ""), key=kp + "cmp", disabled=not anns):
        try:
            with st.spinner("Compiling…"):
                compile_process(m, anns, state, place, query, model, host)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    with st.expander("What these provisions say (plain English)", expanded=rec is None):
        for r in m.itertuples():
            a = anns.get(r.ckey)
            st.markdown(f"**§{section_no(r.header) or '?'} {label(r.header, 70)}** · {r.fn} · {r.topic or '—'}")
            st.write(a["summary"] if a and a["summary"] else clean(r.content)[:240] + "…")
            with st.popover("full text"):
                st.markdown(r.content)

    if rec:
        xml, meta, warns = emit_bpmn(rec, state, place, query, color_dim)
        if xml is None:
            st.warning(" ".join(warns))
            return
        bpmn_viewer(xml, height)
        st.caption("Lanes = who acts. Grey task = no source provision cited (unverified). "
                   + (f"Fill = mean {color_dim.replace('_', ' ')} of the cited provisions." if color_dim else ""))
        for w in warns:
            st.warning(w)
        r1, r2 = st.columns([1, 2])
        status = r1.selectbox("Review status", STATUSES, index=STATUSES.index(rec["status"]), key=kp + "st")
        note = r2.text_input("Reviewer note", rec["note"], key=kp + "nt")
        if (status, note) != (rec["status"], rec["note"]) and st.button("Save review", key=kp + "sv"):
            set_review(pkey, status, note)
            st.rerun()
        fname = f"{state}-{slug(place)}-{slug(query)}"
        d1, d2 = st.columns(2)
        d1.download_button("Download .bpmn", xml, f"{fname}.bpmn", "application/xml", key=kp + "dx")
        d2.download_button("Download sidecar .json", json.dumps({"pkey": pkey, "status": rec["status"], "elements": meta}, indent=2),
                           f"{fname}.json", "application/json", key=kp + "dj")
        with st.expander("Elements and sources"):
            st.dataframe(pd.DataFrame(meta).astype(str))
        wikibase_export_ui(rec, kp)


def grid_cols(n: int, per_row: int) -> list:
    """n column containers laid out in rows of at most per_row."""
    out = []
    for start in range(0, n, per_row):
        out += st.columns(min(per_row, n - start))
    return out


# ───────────────────────────── UI ─────────────────────────────
def main() -> None:
    st.set_page_config(page_title="LOCUS explorer", layout="wide")
    st.title("LOCUS explorer")
    st.caption("Local Ordinance Corpus for the United States (Peskoff et al., arXiv 2606.19334) · "
               "dimension scores are " + SCORES_ARE)

    with st.sidebar:
        st.header("Data")
        src = st.text_input("Parquet source", DEFAULT_SRC, help="Local path, glob, or hf:// glob")
        agg_src = str(SLIM_PATH) if SLIM_PATH.exists() else src
        st.caption(f"National views read: `{agg_src}`")
        if st.button("Build slim cache (no text, fast national views)"):
            cols = ", ".join(["state", "city", "county", "source_jurisdiction_type", '"function"',
                              "topic", "is_substantive", *DIM_LIST])
            with st.spinner("Streaming columns from source — one-time, can take a while…"):
                run(f"COPY (SELECT {cols} FROM {rel(src)}) TO {lit(str(SLIM_PATH))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            st.cache_data.clear()
            st.rerun()
        sub_only = st.checkbox("Substantive chunks only (Rules + Enforcement)", value=True)

        st.header("Ollama")
        host = st.text_input("Host", default_host())
        model = st.text_input("Model", default_model())
        if st.button("Test connection"):
            try:
                names = ollama_models(host)
                st.success(f"Up. Models: {', '.join(names) or 'none pulled'}")
                if model not in names:
                    st.warning(f"`{model}` not found — run `ollama pull {model}`")
            except Exception as e:  # noqa: BLE001
                st.error(f"Not reachable: {e}")
        color_choice = st.selectbox("Colour process boxes by", ["(none)", *DIM_LIST])
        color_dim = None if color_choice == "(none)" else color_choice

    try:
        states = list_states(agg_src)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read `{agg_src}`.\n\n{e}")
        st.info("Check the path/glob, set HF_TOKEN if the dataset is gated, or point LOCUS_SRC at a local parquet.")
        st.stop()

    t_nat, t_jur, t_prov, t_proc, t_cmp = st.tabs(
        ["National view", "Jurisdiction", "Provisions", "Process builder", "Compare"])

    # ── 1. National ──
    with t_nat:
        agg = national_agg(agg_src, sub_only)
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
            fig = px.choropleth(sm, locations=sm["state"].str.upper(), locationmode="USA-states", color=dim,
                                scope="usa", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                hover_data={"n": True}, labels={dim: "mean z"})
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)
        with c2:
            by_type = agg.assign(w=agg[dim] * agg["n"]).groupby("jtype")[["w", "n"]].sum()
            by_type["mean z"] = by_type["w"] / by_type["n"]
            st.plotly_chart(px.bar(by_type.reset_index(), x="jtype", y="mean z", title="Cities vs counties"))
            top = sm.sort_values(dim, ascending=False).head(10)
            st.plotly_chart(px.bar(top, x=dim, y="state", orientation="h", title="Top 10 states"))

        st.subheader("What each kind of jurisdiction regulates")
        comp = composition(agg_src)
        comp = comp[comp["fn"].isin(["Rules", "Enforcement"])] if sub_only else comp
        share = comp.groupby(["jtype", "topic"])["n"].sum().reset_index()
        share["share"] = share["n"] / share.groupby("jtype")["n"].transform("sum")
        st.plotly_chart(px.bar(share, x="jtype", y="share", color="topic", barmode="stack"))
        st.caption("Paper §6: county codes skew toward zoning, city codes toward nuisance/public order.")

    # ── pick a place (tabs 2-4) ──
    with st.sidebar:
        st.header("Jurisdiction")
        state = st.selectbox("State", states, index=states.index("va") if "va" in states else 0)
        plist = list_places(agg_src, state)["place"].tolist()
        place = st.selectbox("City / county", plist,
                             index=plist.index("charlottesville") if "charlottesville" in plist else 0)
    df = load_place(src, state, place) if place else pd.DataFrame()

    # ── 2. Jurisdiction ──
    with t_jur:
        if df.empty:
            st.warning("No rows for that jurisdiction.")
        else:
            m4 = st.columns(4)
            m4[0].metric("Chunks", f"{len(df):,}")
            m4[1].metric("Substantive", f"{df['is_substantive'].mean():.0%}")
            m4[2].metric("Type", ", ".join(sorted(df["jtype"].dropna().unique())))
            m4[3].metric("Mean opacity (z)", f"{df['opacity'].mean():+.2f}")
            a, b = st.columns(2)
            a.plotly_chart(px.histogram(df, x="fn", color="fn", title="Function"))
            b.plotly_chart(px.histogram(df[df["is_substantive"]], x="topic", color="topic", title="Topic (substantive only)"))
            dsel = st.selectbox("Dimension by topic", DIM_LIST, format_func=lambda d: DIMS[d], key="jd")
            st.plotly_chart(px.box(df, x="topic", y=dsel, color="fn", points="outliers"))
            st.plotly_chart(px.scatter(df, x="opacity", y="paternalism", color="topic", hover_name="header", opacity=0.6,
                                       title="Opacity vs paternalism (paper: weakly correlated, r≈0.11 nationally)"))
            strip = df.reset_index().rename(columns={"index": "position"})
            st.plotly_chart(px.scatter(strip, x="position", y="topic", color="fn", hover_name="header",
                                       title="Topic by position in file"))
            st.caption("Assumes file row order = code order. Verify before relying on this view.")
            st.subheader("Hardest to read (highest opacity)")
            st.dataframe(df.sort_values("opacity", ascending=False)[["header", "fn", "topic", "opacity", "enforcement_discretion"]].head(15))

    # ── 3. Provisions ──
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

    # ── 4. Process builder (one jurisdiction) ──
    with t_proc:
        st.markdown(
            "**Annotate → compile → review.** Ollama reads each matching section and extracts actor, action, "
            "condition, deadline, fee, penalty and typed values (stored in SQLite). A second pass assembles those "
            "steps into one process with lanes per actor. Every box cites its source sections; boxes without a "
            "source are grey. The model's output is a draft for a lawyer to check, not a legal reading.")
        c1, c2, c3 = st.columns([3, 1, 1])
        pq = c1.text_input("Topic (all words must appear; stems work)", "dog licen", key="pb_q")
        pcap = c2.number_input("Max sections", 3, 40, 15, key="pb_cap")
        pctx = c3.checkbox("Include Context", False, key="pb_ctx")
        pf = ["Rules", "Process", "Enforcement"] + (["Context"] if pctx else [])
        pm = get_matches(df, pq, pf, int(pcap)) if not df.empty else df
        process_panel(pm, state, place, pq, "pb_", model, host, color_dim)

    # ── 5. Compare ──
    with t_cmp:
        st.markdown("Same topic, several jurisdictions, side by side.")
        c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
        cq = c1.text_input("Topic (all words must appear; stems work)", "dog licen", key="cmp_q")
        ccap = c2.number_input("Max sections each", 3, 40, 15, key="cmp_cap")
        cctx = c3.checkbox("Include Context", False, key="cmp_ctx")
        n = int(c4.number_input("Places to compare", 2, 8, 2, key="cmp_n"))
        per_row = int(c5.number_input("Columns per row", 1, 4, 3, key="cmp_cols"))
        cf = ["Rules", "Process", "Enforcement"] + (["Context"] if cctx else [])

        picks: list[tuple[str, str]] = []
        for i, col in enumerate(grid_cols(n, per_row)):
            with col:
                dstate = state if state in states else states[0]
                s = st.selectbox("State", states, index=states.index(dstate), key=f"cs{i}")
                pl = list_places(agg_src, s)["place"].tolist()
                prefer = ["charlottesville", "albemarle"][i] if i < 2 else None
                idx = pl.index(prefer) if prefer in pl else min(i, max(len(pl) - 1, 0))
                picks.append((s, st.selectbox("City / county", pl, index=idx, key=f"cp{i}")))

        data = []
        for s, p in picks:
            d = load_place(src, s, p)
            mm = get_matches(d, cq, cf, int(ccap))
            data.append((s, p, mm, get_annotations(mm["ckey"].tolist(), model) if len(mm) else {}))
        table = pd.DataFrame([{"Jurisdiction": f"{p.title()}, {s.upper()}", **summarize(mm, an)} for s, p, mm, an in data])
        table["Jurisdiction"] = table["Jurisdiction"] + table.groupby("Jurisdiction").cumcount().map(lambda i: f" ({i + 1})" if i else "")
        st.subheader("At a glance")
        st.dataframe(table.set_index("Jurisdiction").T.astype(str))

        dim_cmp = st.selectbox("Compare distribution of", DIM_LIST, format_func=lambda d: DIMS[d], key="cmp_dim")
        frames = [mm.assign(J=f"{p.title()}, {s.upper()}") for s, p, mm, _ in data if len(mm)]
        if frames:
            st.plotly_chart(px.strip(pd.concat(frames, ignore_index=True), x="J", y=dim_cmp, color="fn", hover_name="header"))

        st.subheader("Provisions and process models")
        for i, (col, (s, p, mm, an)) in enumerate(zip(grid_cols(n, per_row), data)):
            with col:
                st.markdown(f"### {p.title()}, {s.upper()}")
                process_panel(mm, s, p, cq, f"cmp{i}_", model, host, color_dim, height=460)


if __name__ == "__main__":
    main()
