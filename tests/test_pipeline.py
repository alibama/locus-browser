import re
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pytest

import lexipedia_export as X
import locus_core as C
from locus_bpmn import emit_bpmn

NS = {"b": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
APP = str(Path(__file__).resolve().parents[1] / "locus_explorer.py")


def load(pq, state, place):
    return duckdb.sql(f"""SELECT header, content, is_substantive, "function" fn, topic, source_jurisdiction_type jtype, state, city, county,
        opacity, enforcement_discretion, paternalism, problem_salience FROM '{pq}'
        WHERE state='{state}' AND coalesce(city, county)='{place}'""").df()


def run_place(env, place, query="dog licen"):
    df = load(env["pq"], "va", place)
    m = C.get_matches(df, query, ["Rules", "Process", "Enforcement"], 15)
    todo = m[~m["ckey"].isin(C.get_annotations(m["ckey"].tolist(), env["model"]))]
    assert not C.annotate(todo, env["model"], env["host"])
    anns = C.get_annotations(m["ckey"].tolist(), env["model"])
    pkey = C.compile_process(m, anns, "va", place, query, env["model"], env["host"])
    return m, anns, pkey


def test_grounded_num():
    assert C.grounded_num("10", "The fee is $10 per dog") == "10"
    assert C.grounded_num("80", "The annual license fee is $8.") == ""
    assert C.grounded_num("1,000", "a fine of $1,000") == "1000"
    assert C.grounded_num("", "x") == ""


def test_annotation_typed_and_grounded(env):
    _, anns, _ = run_place(env, "albemarle")
    steps = [s for a in anns.values() for s in a["steps"]]
    assert steps and all(s["fee_usd"] == "" for s in steps)            # hallucinated "80" was dropped
    m2, anns2, _ = run_place(env, "charlottesville")
    assert "10" in {s["fee_usd"] for a in anns2.values() for s in a["steps"]}
    assert any(s["deadline_days"] == "30" for a in anns2.values() for s in a["steps"])
    assert len(m2) == 3                                               # barking section excluded by 'licen'


def test_bpmn_wellformed_and_deterministic(env):
    _, _, pkey = run_place(env, "charlottesville")
    rec = C.get_process(pkey)
    xml, meta, warns = emit_bpmn(rec, "va", "charlottesville", "dog licen", "opacity")
    assert xml == emit_bpmn(rec, "va", "charlottesville", "dog licen", "opacity")[0]
    root = ET.fromstring(xml)
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    for f in root.iter("{%s}sequenceFlow" % NS["b"]):
        assert f.get("sourceRef") in ids and f.get("targetRef") in ids
    lanes = [l.get("name") for l in root.iter("{%s}lane" % NS["b"])]
    assert {"Dog owner", "Treasurer", "Court"} <= set(lanes)
    assert any("Unsourced" in w for w in warns) and any("loop-back" in w for w in warns)
    assert any(m.get("fee_usd") == "10" for m in meta)


def test_export_gate_and_two_pass(env):
    _, _, pkey = run_place(env, "charlottesville")
    props = X.load_props()
    with pytest.raises(PermissionError):
        X.export_files(C.get_process(pkey), "Dog licensing", {}, props)
    C.set_review(pkey, "reviewed-ok", "checked by test")
    rec = C.get_process(pkey)

    bundle, p1, p2, rep = X.export_files(rec, "Dog licensing", {}, props)
    assert any("has no id yet" in w for w in rep["pass1"]["warnings"])
    assert "class:norm" in rep["pass1"]["unresolved"]

    for i, k in enumerate(k for k, p in props["properties"].items() if not p["id"]):
        props["properties"][k]["id"] = f"P{100 + i}"
    qm = {k: f"Q{1000 + i}" for i, k in enumerate(props["externals"])}
    qm[bundle["jurisdiction"]["key"]] = "Q999"
    bundle, p1, p2, rep = X.export_files(rec, "Dog licensing", qm, props)
    assert not rep["pass1"]["unresolved"]
    assert p1.count("CREATE") == len(bundle["items"])
    for line in p1.splitlines():
        assert line == "CREATE" or line.count("\t") >= 2
    assert re.search(r"LAST\tP\d+\t\+10U1\d+", p1)                      # fee 10 USD as a quantity
    assert re.search(r"LAST\tP\d+\t\+30U1\d+", p1)                      # 30 days

    qm2 = {**qm, **{it["key"]: f"Q{5000 + i}" for i, it in enumerate(bundle["items"])}}
    _, p1b, p2b, rep2 = X.export_files(rec, "Dog licensing", qm2, props)
    assert p1b == "" and not rep2["pass2"]["missing_qid"]
    assert any(l.split("\t")[1] == props["properties"]["has_part"]["id"] for l in p2b.splitlines())
    fb = [l for l in p2b.splitlines() if l.split("\t")[1] == props["properties"]["followed_by"]["id"]]
    assert fb and any(len(l.split("\t")) == 5 for l in fb)              # a followed-by with a condition qualifier


def test_app_compare_n_places(env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    cp = lambda: [s.key for s in at.selectbox if s.key and s.key.startswith("cp")]
    at.number_input(key="cmp_n").set_value(3).run()
    assert not at.exception and cp() == ["cp0", "cp1", "cp2"]
    at.number_input(key="cmp_n").set_value(5).run()
    assert len(cp()) == 5
    at.number_input(key="cmp_n").set_value(2).run()
    for i in (0, 1):
        at.button(key=f"cmp{i}_ann").click().run()
        at.button(key=f"cmp{i}_cmp").click().run()
        assert not at.exception and not at.error
    table = [d.value for d in at.dataframe if "Fee (USD)" in d.value.index]
    assert table and "$10" in table[0].loc["Fee (USD)"].tolist()[0]
