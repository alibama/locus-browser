import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pytest

import fake_ollama
import lexipedia_export as X
import locus_core as C
import locus_markup as M
import locus_sources as S
from locus_bpmn import emit_bpmn
from locus_dmn import decision_xml
from locus_llm import LLMUnavailable, call_llm, model_available, model_selftest, pull_model, server_status, start_server

NS = {"b": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
APP = str(Path(__file__).resolve().parents[1] / "locus_explorer.py")
LOCAL_GOLD = None


def local(env, place="charlottesville"):
    return S.load_locus_place(duckdb.connect().cursor(), str(env["pq"]), "va", place)


def candidates(env, query, place="charlottesville", state_law=True):
    cur = duckdb.connect().cursor()
    frames = {"local": local(env, place)}
    if state_law:
        frames["state"], _ = S.load_law_context(cur, env["template"], "va", ["statutes"], query.split(), 6, [])
    return C.build_candidates(frames, query, {"local": 15, "state": 6, "federal": 0}, ["Rules", "Process", "Enforcement", "Context"])


def job(env, m, query, model="qwen2.5:7b", place="charlottesville", verifier="", workers=1):
    return M.Job(state="va", place=place, query=query, chunks=m, model=model, host=env["host"], verifier=verifier, workers=workers)


# ───────────────────────── data merge ─────────────────────────
def test_grounded_num():
    assert C.grounded_num("10", "The fee is $10 per dog") == "10"
    assert C.grounded_num("80", "The annual license fee is $8.") == ""
    assert C.grounded_num("1,000", "a fine of $1,000") == "1000"
    assert C.grounded_num("0.36", "36 cents per hundred dollars") == "0.36"


def test_open_us_law_adapter(env):
    cur = duckdb.connect().cursor()
    df = S.search_oul(cur, str(env["oul"]), "va", ["license", "tax"], 10)
    assert list(df["citation"]) == ["Va. Code § 58.1-3703.1"]            # repealed 1-1 and cattle excluded
    r = df.iloc[0]
    assert r["level"] == "state" and r["source"] == "open-us-law" and r["corpus"] == "statutes" and r["place"] == "statewide"
    assert r["section_no"] == "58.1-3703.1" and r["header"].startswith("Va. Code § 58.1-3703.1 Local license taxes")
    assert set(S.UNIFIED) <= set(df.columns)
    pinned = S.search_oul(cur, str(env["oul"]), "va", ["zzz"], 10, pinned=["3-9"])
    assert list(pinned["section_no"]) == ["3-9"]
    df2, problems = S.load_law_context(cur, env["template"], "va", ["statutes", "regulations"], ["license"], 5, [])
    assert len(df2) == 1 and any("regulations" in p for p in problems)    # missing file is reported, not fatal
    loc = local(env)
    assert loc["level"].eq("local").all() and loc["source"].eq("locus").all() and set(S.UNIFIED) <= set(loc.columns)


# ───────────────────────── model server ─────────────────────────
def test_model_server_status_pull_and_missing(env):
    st = server_status(env["host"])
    assert st["up"] and model_available(st, "qwen2.5:7b") and not model_available(st, "llama3:8b")
    assert not server_status("http://127.0.0.1:1", 0.5)["up"]
    with pytest.raises(LLMUnavailable, match="not available"):
        call_llm("llama3:8b", env["host"], "s", "u", {"type": "object"})
    seen = [d["status"] for d in pull_model(env["host"], "llama3:8b")]
    assert seen[-1] == "success" and model_available(server_status(env["host"]), "llama3:8b")
    assert model_selftest(env["host"], "llama3:8b")["ok"]
    ok, msg = start_server("http://remote.example:11434")
    assert not ok and "not local" in msg
    with pytest.raises(LLMUnavailable, match="Cannot reach"):
        call_llm("x", "http://127.0.0.1:1", "s", "u", {"type": "object"})


# ───────────────────────── the whole pipeline ─────────────────────────
def test_business_pipeline_end_to_end(env):
    m = candidates(env, "business licen")
    assert set(m["level"]) == {"local", "state"}
    j = job(env, m, "business licen")
    prog = M.Progress()
    log = M.run_pipeline(j, prog)
    assert not [e for e in log["errors"] if "unusable" in e], log["errors"]
    stt = M.pipeline_status(j)
    keys = m["ckey"].tolist()
    tri, rel = stt["triage_data"], stt["relevance_data"]

    # artifact chunk never reached the LLM; off-topic and admin chunks were set aside with a reason
    art = m[m["header"].str.contains("LICENSES")]
    if len(art):
        k = art.iloc[0]["ckey"]
        assert tri[k]["role"] == "heading_or_artifact" and tri[k].get("skipped_llm") and rel[k]["verdict"] == "no"
    by_title = {r["header"]: r["ckey"] for _, r in m.iterrows()}
    veh = next(k for t, k in by_title.items() if "inoperable" in t)
    insp = next(k for t, k in by_title.items() if "inspector" in t)
    state = next(k for t, k in by_title.items() if "3703.1" in t)
    assert rel[veh]["verdict"] == "no" and rel[insp]["verdict"] == "no" and rel[state]["verdict"] == "partial"
    assert tri[state]["role"] == "authority"

    # new subject proposed, not silently accepted
    assert C.subjects()["brand-new-thing"]["status"] == "proposed"

    rec = C.get_process(stt["pkey"])
    assert rec and rec["regime"]["missing"] and rec["decisions"] and rec["rules"]
    assert any(r["rule_type"] == "threshold" and r["threshold_value"] == "100000" for r in rec["rules"])   # state ceiling captured

    # rate table became a decision table with rates typed as percent, never as a flat fee
    rows = rec["decisions"][0]["rows"]
    assert {r["amount_kind"] for r in rows} == {"percent_of_base", "flat_fee"}
    assert next(r for r in rows if r["class_label"] == "Class IV")["amount_value"] == "0.36"
    dmn = decision_xml(rec["decisions"][0], "charlottesville", "va")
    root = ET.fromstring(dmn)
    assert len(root.findall(".//{*}rule")) == 3 and "&lt;=50000" in dmn
    assert "&gt;=100000" not in dmn          # the model's 100000 is not in THIS section's text (it's in the state law) -> dropped

    # verification: fabricated quote is downgraded, real quote is supported
    v = rec["verdicts"]
    assert v and any(x["verdict"] == "supported" for x in v.values())
    assert all(x["quote_ok"] or x["verdict"] != "supported" for x in v.values())

    xml, meta, warns = emit_bpmn(rec, "va", "charlottesville", "business licen", "opacity")
    assert xml == emit_bpmn(rec, "va", "charlottesville", "business licen", "opacity")[0]
    root = ET.fromstring(xml)
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    assert all(f.get("sourceRef") in ids and f.get("targetRef") in ids for f in root.iter("{%s}sequenceFlow" % NS["b"]))
    assert root.find(".//b:businessRuleTask", NS) is not None
    assert any("Unsourced" in w for w in warns)
    # end nodes never carry facts
    assert not any(x["type"] == "end" and ("amount_value" in x or "deadline_days" in x) for x in meta)

    # re-running is free: no new LLM calls
    n = len(fake_ollama.STATE["calls"])
    M.run_pipeline(j, M.Progress())
    assert len(fake_ollama.STATE["calls"]) == n


def test_flat_fee_grounding_and_rule_typing(env):
    m = candidates(env, "dog licen", place="albemarle", state_law=False)
    j = job(env, m, "dog licen", place="albemarle")
    M.run_pipeline(j, M.Progress())
    ex = C.markup_get(m["ckey"].tolist(), "extract", j.model)
    rules = [r for e in ex.values() for r in e.get("rules", [])]
    fee = next(r for r in rules if "application" in r["action"])
    assert fee["amount_value"] == "" and fee["amount_kind"] == "none"     # hallucinated "80" dropped; unverifiable flat fee is not a fee
    assert next(r for r in rules if r["rule_type"] == "duty")["renewal"] == "annual"


def test_unsupported_verdict_marks_the_node(env, monkeypatch):
    m = candidates(env, "dog licen", state_law=False)
    j = job(env, m, "dog licen")
    real = M.call_llm

    def bad(model, host, system, user, schema, **kw):
        if kw.get("stage") == "verify":
            return {"verdict": "unsupported", "issues": ["text does not say this"], "quote": ""}
        return real(model, host, system, user, schema, **kw)

    monkeypatch.setattr(M, "call_llm", bad)
    M.run_pipeline(j, M.Progress())
    rec = C.get_process(M.pipeline_status(j)["pkey"])
    xml, meta, warns = emit_bpmn(rec, "va", "charlottesville", "dog licen", None)
    assert any("Unsupported by its source" in w for w in warns)
    assert 'bioc:stroke="#7f1d1d"' in xml
    keep, held = X.exportable(rec)
    assert not keep and any("unsupported" in h for h in held)


# ───────────────────────── two models at the same time ─────────────────────────
def test_two_models_run_concurrently_and_stay_separate(env):
    m = candidates(env, "dog licen", state_law=False)
    ja, jb = job(env, m, "dog licen", "qwen2.5:7b"), job(env, m, "dog licen", "tiny")
    fake_ollama.STATE["delay"] = 0.05
    res = M.run_jobs([ja, jb], M.Progress(), parallel=True)
    assert fake_ollama.STATE["max_models_inflight"] == 2                  # both models were in flight together
    assert res[ja.pid]["pkey"] != res[jb.pid]["pkey"]
    ra, rb = C.get_process(res[ja.pid]["pkey"]), C.get_process(res[jb.pid]["pkey"])
    assert ra["model"] == "qwen2.5:7b" and rb["model"] == "tiny"
    ag = M.agreement(ra, rb)
    assert ag["typed_fact_overlap"] is not None and ag["typed_fact_overlap"] < 1   # 'tiny' dropped the fee

    fake_ollama.reset()
    jc, jd = job(env, m, "dog licen", "qwen2.5:14b"), job(env, m, "dog licen", "tiny")
    fake_ollama.STATE["delay"] = 0.05
    M.run_jobs([jc, jd], M.Progress(), parallel=False)
    assert fake_ollama.STATE["max_models_inflight"] == 1                  # sequential means one at a time


def test_workers_per_model(env):
    m = candidates(env, "dog licen", state_law=False)
    fake_ollama.STATE["delay"] = 0.05
    M.run_pipeline(job(env, m, "dog licen", "qwen2.5:14b", workers=3), M.Progress())
    assert fake_ollama.STATE["max_inflight"] >= 2


def test_unreachable_server_stops_cleanly(env):
    m = candidates(env, "dog licen", state_law=False)
    j = M.Job(state="va", place="charlottesville", query="dog licen", chunks=m, model="qwen2.5:7b", host="http://127.0.0.1:1")
    prog = M.Progress()
    log = M.run_pipeline(j, prog)
    assert any("Cannot reach" in e for e in log["errors"]) and prog.snapshot()[j.pid]["stage"] == "stopped"


# ───────────────────────── export ─────────────────────────
def test_export_gate_subjects_and_two_pass(env):
    m = candidates(env, "dog licen", state_law=False)
    j = job(env, m, "dog licen", verifier="qwen2.5:14b")
    M.run_pipeline(j, M.Progress())
    pkey = M.pipeline_status(j)["pkey"]
    props = X.load_props()
    with pytest.raises(PermissionError, match="reviewed-ok"):
        X.export_files(C.get_process(pkey), "dog-license", {}, props)
    C.set_review(pkey, "reviewed-ok", "", "")
    with pytest.raises(PermissionError, match="reviewer"):
        X.export_files(C.get_process(pkey), "dog-license", {}, props)
    C.set_review(pkey, "reviewed-ok", "checked every step against the code", "R. Hubbard")
    rec = C.get_process(pkey)
    with pytest.raises(ValueError, match="registry"):
        X.build_bundle({**rec, "subject": "", "query_interp": {}}, "not-a-subject")

    bundle, p1, p2, rep = X.export_files(rec, "dog-license", {}, props)
    assert bundle["subject"]["label"] == "Dog license"                     # controlled label, not the LLM's title
    assert any("Issue license" in h for h in rep["held_back"])             # fabricated-quote step held back
    assert "Issue license" not in json.dumps(bundle["items"])
    bundle_p, *_ = X.export_files(rec, "dog-license", {}, props, include_partial=True)
    assert "Issue license" in json.dumps(bundle_p["items"])

    for i, k in enumerate(k for k, p in props["properties"].items() if not p["id"]):
        props["properties"][k]["id"] = f"P{100 + i}"
    qm = {k: f"Q{1000 + i}" for i, k in enumerate(props["externals"])}
    qm[bundle["jurisdiction"]["key"]] = "Q999"
    bundle, p1, p2, rep = X.export_files(rec, "dog-license", qm, props)
    assert not rep["pass1"]["unresolved"] and p1.count("CREATE") == len(bundle["items"])
    assert re.search(r"LAST\tP\d+\t\+10U1\d+", p1) and re.search(r"LAST\tP\d+\t\+30U1\d+", p1)
    qm2 = {**qm, **{it["key"]: f"Q{5000 + i}" for i, it in enumerate(bundle["items"])}}
    _, p1b, p2b, rep2 = X.export_files(rec, "dog-license", qm2, props)
    assert p1b == "" and not rep2["pass2"]["missing_qid"] and "\n" in p2b


def test_rate_is_exported_as_percent_never_flat_fee(env):
    m = candidates(env, "business licen", state_law=False)
    j = job(env, m, "business licen", verifier="qwen2.5:14b")
    M.run_pipeline(j, M.Progress())
    rec = C.get_process(M.pipeline_status(j)["pkey"])
    # make the decision-table task a real rate rule so the exporter has something typed to map
    rec["rules"].append({**rec["rules"][0], "code": "c9.1", "chunk": rec["rules"][0]["chunk"], "amount_kind": "percent_of_base",
                         "amount_value": "0.36", "amount_unit": "usd_per_100", "amount_base": "gross receipts"})
    node = next(n for n in rec["graph"]["nodes"] if n["type"] == "task")
    node["sources"] = ["c9.1"]
    rec["verdicts"]["c9.1"] = {"verdict": "supported"}
    rec.update(status="reviewed-ok", reviewer="x", note="y")
    b = X.build_bundle(rec, "business-license")
    stmts = [s for it in b["items"] for s in it["statements"]]
    assert any(s["prop"] == "rate_percent" and s["amount"] == 0.36 for s in stmts)
    assert not any(s["prop"] == "fee_amount" and s["amount"] == 0.36 for s in stmts)


# ───────────────────────── the app ─────────────────────────
def test_app_places_models_and_pull(env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("Ollama is up" in s.value for s in at.success)
    cp = lambda: [s.key for s in at.selectbox if s.key and s.key.startswith("cp")]
    at.number_input(key="cmp_n").set_value(3).run()
    assert not at.exception and cp() == ["cp0", "cp1", "cp2"]
    at.number_input(key="cmp_n").set_value(5).run()
    assert len(cp()) == 5
    at.number_input(key="cmp_n").set_value(2).run()

    # Model B that is not installed -> pull from the UI
    at.text_input(key="model_b").set_value("llama3:8b").run()
    assert any("not installed" in md.value for md in at.markdown)
    at.button(key="pull1").click().run()
    assert not at.exception
    assert server_status(env["host"])["up"] and model_available(server_status(env["host"]), "llama3:8b")

    # two models, run everything in parallel from the Compare tab
    at.text_input(key="model_b").set_value("qwen2.5:14b").run()
    at.text_input(key="cmp_q").set_value("dog licen").run()
    at.button(key="cmp_all").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert fake_ollama.STATE["max_models_inflight"] == 2
    glance = next(d.value for d in at.dataframe if "Flat fee (USD)" in d.value.index)
    assert any("$10" in str(v) for v in glance.loc["Flat fee (USD)"].tolist())
    assert any("qwen2.5:14b" in c for c in glance.columns)


def test_app_server_down_is_explained(env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[[t.label for t in at.text_input].index("Ollama host")].set_value("http://127.0.0.1:1").run()
    assert any("not reachable" in e.value for e in at.error)
    assert any(b.label == "Start Ollama here" for b in at.button)
