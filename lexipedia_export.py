"""
lexipedia_export — reviewed process record -> Wikibase item bundle -> QuickStatements (dry run).

Nothing here writes to a Wikibase. It emits files; you load them with your own dry-run-then-confirm step.

Two passes, matching the existing Lexipedia pipeline:
  pass 1  CREATE items with labels, descriptions, and every statement whose value is a literal or an
          already-known external item (classes, units, modalities, jurisdiction).
  pass 2  after you record the QIDs pass 1 produced in qid_map.json (local key -> QID), emit the
          statements that link new items to each other (has part, based on, followed by, bearer, ...).

    python lexipedia_export.py check
    python lexipedia_export.py list
    python lexipedia_export.py export --pkey <key> --subject "Dog licensing" [--qid-map qid_map.json] [--out out/]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

from locus_core import get_process, label, list_processes, node_facts, normalize_graph, slug

HERE = Path(__file__).parent


def load_props(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or HERE / "properties.json").read_text())


# ───────────────────────────── bundle ─────────────────────────────
def _title(ch: dict) -> str:
    t = label(ch["header"], 120)
    if ch["section"]:
        t = re.sub(rf"^(?:sec(?:tion|\.)?\s*)?§?\s*{re.escape(ch['section'])}[\s.\-–:]*", "", t, flags=re.I)
    return t


def _s(prop, kind, **kw):
    return {"prop": prop, "kind": kind, **kw}


def build_bundle(rec: dict, subject_label: str) -> dict:
    state, place = rec["state"], rec["place"]
    PL = f"{place.title()}, {state.upper()}"
    chunks = {c["code"]: c for c in rec["chunks"]}
    steps_by_code = {s["code"]: s for s in rec.get("steps", [])}
    nodes, order, edges, _ = normalize_graph(rec["graph"], set(chunks), set(steps_by_code))
    tasks = [n for n in order if nodes[n]["type"] == "task"]
    pk = rec.get("pkey") or slug(f"{state}-{place}-{rec.get('query', '')}")
    jur, subj = f"jur:{state}/{slug(place)}", f"subject:{slug(subject_label)}"
    record_text = f"LOCUS-v1 + {rec.get('model', '?')} prompt {rec.get('prompt_ver', '?')}; review: {rec.get('status', '?')} {rec.get('note', '')}".strip()

    items: list[dict] = []

    def item(key, lab, desc, stmts):
        items.append({"key": key, "label": lab, "description": desc, "aliases": [f"lex-local:{key}"], "statements": stmts})

    item(subj, subject_label, "regulated subject", [_s("instance_of", "ext", key="class:subject", text="regulated subject")])

    used_chunks: list[str] = []
    for n in tasks:
        for c in nodes[n]["sources"]:
            if c not in used_chunks:
                used_chunks.append(c)
    prov_key = {c: f"prov:{chunks[c]['key']}" for c in used_chunks}
    for c in used_chunks:
        ch = chunks[c]
        st = [_s("instance_of", "ext", key="class:provision", text="ordinance provision"),
              _s("jurisdiction", "ext", key=jur, text=PL),
              _s("regulated_subject", "local", key=subj),
              _s("citation", "string", value=f"{PL}, {'§ ' + ch['section'] + ' ' if ch['section'] else ''}{_title(ch)}"),
              _s("source_chunk_id", "string", value=ch["key"])]
        if ch["section"]:
            st.append(_s("legal_identifier", "string", value=ch["section"]))
        item(prov_key[c], f"{PL} § {ch['section'] or label(ch['header'], 40)}", f"ordinance provision, {PL}", st)

    canon: dict[str, str] = {}
    for n in tasks:
        a = nodes[n]["actor"]
        if a:
            canon.setdefault(a.lower(), a)
    for low, a in canon.items():
        item(f"actor:{slug(a)}", a, "legal role", [_s("instance_of", "ext", key="class:legal-role", text="legal role")])

    norm_key = {n: f"norm:{pk}:{slug(n)}" for n in tasks}
    for n in tasks:
        nd = nodes[n]
        f = node_facts(nd, steps_by_code)
        st = [_s("instance_of", "ext", key="class:norm", text="legal norm"),
              _s("jurisdiction", "ext", key=jur, text=PL),
              _s("regulated_subject", "local", key=subj)]
        if f.get("modality"):
            st.append(_s("deontic_modality", "ext", key=f"modality:{f['modality']}", text=f["modality"]))
        if nd["actor"]:
            st.append(_s("bearer", "local", key=f"actor:{slug(canon[nd['actor'].lower()])}"))
        if f.get("condition"):
            st.append(_s("condition", "string", value=f["condition"]))
        if f.get("penalty"):
            st.append(_s("sanction", "string", value=f["penalty"]))
        if f.get("fee_usd"):
            st.append(_s("fee_amount", "quantity", amount=float(f["fee_usd"]), unit="unit:usd"))
        if f.get("deadline_days"):
            st.append(_s("time_limit", "quantity", amount=float(f["deadline_days"]), unit="unit:day"))
        if f.get("renewal"):
            st.append(_s("renewal_interval", "string", value=f["renewal"]))
        if f.get("penalty_max_usd"):
            st.append(_s("max_penalty_amount", "quantity", amount=float(f["penalty_max_usd"]), unit="unit:usd"))
        st += [_s("based_on", "local", key=prov_key[c]) for c in nd["sources"]]
        item(norm_key[n], f"{nd['label']} ({PL})", f"step in {subject_label} process, {PL}", st)

    # task -> next task, looking through gateways; the gateway question/answer becomes a qualifier
    out_edges: dict[str, list[dict]] = {}
    for e in edges:
        out_edges.setdefault(e["src"], []).append(e)
    by_key = {i["key"]: i for i in items}
    for t in tasks:
        seen: set[str] = set()
        stack = [(e["tgt"], []) for e in out_edges.get(t, [])]
        while stack:
            v, path = stack.pop(0)
            if v in seen:
                continue
            seen.add(v)
            if nodes[v]["type"] == "task":
                q = [{"prop": "condition", "kind": "string", "value": " / ".join(path)}] if path else []
                by_key[norm_key[t]]["statements"].append(_s("followed_by", "local", key=norm_key[v], qualifiers=q))
            elif nodes[v]["type"] == "gateway":
                for e in out_edges.get(v, []):
                    stack.append((e["tgt"], path + [f"{nodes[v]['label']} → {e['label']}" if e["label"] else nodes[v]["label"]]))

    proc = [_s("instance_of", "ext", key="class:process", text="legal process model"),
            _s("jurisdiction", "ext", key=jur, text=PL),
            _s("regulated_subject", "local", key=subj),
            _s("extraction_record", "string", value=record_text)]
    proc += [_s("based_on", "local", key=prov_key[c]) for c in used_chunks]
    proc += [_s("has_part", "local", key=norm_key[n]) for n in tasks]
    item(f"proc:{pk}", f"{rec['graph'].get('title') or subject_label} — {PL}", "legal process model (LLM-assisted, human-reviewed)", proc)
    return {"jurisdiction": {"key": jur, "label": PL}, "subject": {"key": subj, "label": subject_label}, "items": items}


# ───────────────────────────── QuickStatements rendering ─────────────────────────────
def _str(v: str) -> str:
    return '"' + re.sub(r"\s+", " ", str(v).replace('"', "'")).strip() + '"'


def _num(q: str) -> str:
    return re.sub(r"\D", "", q)


def _value(stmt: dict, qid_map: dict, props: dict, rep: dict):
    """-> (status, token); status ∈ ok | defer | unresolved | skip"""
    p = props["properties"].get(stmt["prop"], {})
    dt, kind = p.get("datatype"), stmt["kind"]
    if not p.get("id"):
        rep["warnings"].add(f"property '{stmt['prop']}' has no id yet — statement skipped")
        return "skip", None
    if kind == "local":
        q = qid_map.get(stmt["key"])
        return ("ok", q) if q else ("defer", None)
    if kind == "ext":
        if dt == "wikibase-item":
            q = qid_map.get(stmt["key"])
            if q:
                return "ok", q
            rep["unresolved"].add(stmt["key"])
            return "unresolved", None
        return "ok", _str(stmt["text"])
    if kind == "string":
        if dt in ("string", "external-id"):
            return "ok", _str(stmt["value"])
        if dt == "monolingualtext":
            return "ok", "en:" + _str(stmt["value"])
        rep["warnings"].add(f"'{stmt['prop']}' is {dt}, not text — string value skipped")
        return "skip", None
    if kind == "quantity":
        if dt != "quantity":
            rep["warnings"].add(f"'{stmt['prop']}' is {dt}, not quantity — skipped")
            return "skip", None
        u = qid_map.get(stmt["unit"])
        if not u:
            rep["unresolved"].add(stmt["unit"])
            return "unresolved", None
        a = stmt["amount"]
        return "ok", f"+{int(a) if a == int(a) else a}U{_num(u)}"
    return "skip", None


def _quals(stmt: dict, qid_map: dict, props: dict, rep: dict) -> str:
    out = ""
    for q in stmt.get("qualifiers", []):
        st, tok = _value(q, qid_map, props, rep)
        if st == "ok":
            out += f"\t{props['properties'][q['prop']]['id']}\t{tok}"
    return out


def render_pass1(bundle: dict, qid_map: dict, props: dict):
    rep = {"unresolved": set(), "warnings": set(), "created": [], "deferred": 0}
    lines: list[str] = []
    for it in bundle["items"]:
        if it["key"] in qid_map:
            continue
        rep["created"].append(it["key"])
        lines += ["CREATE", f"LAST\tLen\t{_str(it['label'])}", f"LAST\tDen\t{_str(it['description'])}"]
        lines += [f"LAST\tAen\t{_str(a)}" for a in it["aliases"]]
        for stmt in it["statements"]:
            st, tok = _value(stmt, qid_map, props, rep)
            if st == "ok":
                lines.append(f"LAST\t{props['properties'][stmt['prop']]['id']}\t{tok}{_quals(stmt, qid_map, props, rep)}")
            elif st == "defer":
                rep["deferred"] += 1
    return "\n".join(lines) + ("\n" if lines else ""), rep


def render_pass2(bundle: dict, qid_map: dict, props: dict):
    rep = {"unresolved": set(), "warnings": set(), "missing_qid": set()}
    lines: list[str] = []
    for it in bundle["items"]:
        subj = qid_map.get(it["key"])
        for stmt in it["statements"]:
            if stmt["kind"] != "local":
                continue
            if not subj:
                rep["missing_qid"].add(it["key"])
                continue
            st, tok = _value(stmt, qid_map, props, rep)
            if st == "ok":
                lines.append(f"{subj}\t{props['properties'][stmt['prop']]['id']}\t{tok}{_quals(stmt, qid_map, props, rep)}")
            elif st == "defer":
                rep["missing_qid"].add(stmt["key"])
    return "\n".join(lines) + ("\n" if lines else ""), rep


def export_files(rec: dict, subject: str, qid_map: dict, props: dict, allow_draft: bool = False):
    if rec.get("status") != "reviewed-ok" and not allow_draft:
        raise PermissionError(f"Process status is '{rec.get('status')}'; mark it reviewed-ok first.")
    bundle = build_bundle(rec, subject)
    p1, r1 = render_pass1(bundle, qid_map, props)
    p2, r2 = render_pass2(bundle, qid_map, props)
    report = {"pass1": {k: sorted(v) if isinstance(v, set) else v for k, v in r1.items()},
              "pass2": {k: sorted(v) for k, v in r2.items()}}
    return bundle, p1, p2, report


# ───────────────────────────── live check + CLI ─────────────────────────────
def check_properties(props: dict, timeout: int = 15) -> list[str]:
    ids = {p["id"]: (k, p) for k, p in props["properties"].items() if p.get("id")}
    url = f"{props['api']}?action=wbgetentities&ids={'|'.join(ids)}&props=labels|datatype&languages=en&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "locus-explorer property-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    out = []
    for pid, (k, p) in ids.items():
        e = data.get("entities", {}).get(pid)
        if not e or "missing" in e:
            out.append(f"{pid} ({k}): not found on the instance")
            continue
        live_dt, live_lab = e.get("datatype"), e.get("labels", {}).get("en", {}).get("value")
        if live_dt != p["datatype"]:
            out.append(f"{pid} ({k}): datatype is {live_dt}, config says {p['datatype']}")
        if live_lab and live_lab != p["label"]:
            out.append(f"{pid} ({k}): label is '{live_lab}', config says '{p['label']}'")
    return out or ["all configured properties match the instance"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="compare properties.json with the live instance")
    sub.add_parser("list", help="list stored processes")
    ex = sub.add_parser("export", help="write bundle.json, pass1.qs, pass2.qs, report.json")
    ex.add_argument("--pkey", required=True)
    ex.add_argument("--subject", required=True)
    ex.add_argument("--qid-map")
    ex.add_argument("--props")
    ex.add_argument("--out", default="out")
    ex.add_argument("--allow-draft", action="store_true", help="testing only")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for p in list_processes():
            print(p["pkey"], p["state"], p["place"], repr(p["query"]), p["status"])
        return 0
    props = load_props(getattr(a, "props", None))
    if a.cmd == "check":
        try:
            print("\n".join(check_properties(props)))
        except Exception as e:  # noqa: BLE001
            print(f"Could not reach {props['api']}: {e}\n(If the site sits behind a bot challenge, run this from a browser-trusted network "
                  "or compare the properties manually.)", file=sys.stderr)
            return 1
        return 0
    rec = get_process(a.pkey)
    if rec is None:
        print("no such process", file=sys.stderr)
        return 1
    qm = json.loads(Path(a.qid_map).read_text()) if a.qid_map else {}
    try:
        bundle, p1, p2, report = export_files(rec, a.subject, qm, props, a.allow_draft)
    except PermissionError as e:
        print(e, file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    (out / "pass1.qs").write_text(p1)
    (out / "pass2.qs").write_text(p2)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{len(bundle['items'])} items · pass1 {p1.count(chr(10))} lines · pass2 {p2.count(chr(10))} lines")
    for k in ("unresolved", "warnings"):
        for v in report["pass1"][k]:
            print(f"  [{k}] {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
