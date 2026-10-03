"""
Deterministic lint for a compiled process + its exports. No LLM, no network.

    python evals/lint_process.py --bpmn x.bpmn --sidecar x.json [--bundle b.json] [--pass1 p1.qs] [--gold evals/gold/x.json]

It cannot say whether a process is *right*. It finds the failure patterns we have already seen:
off-topic provisions, steps invented from near-empty chunks, rates stored as fees, no branching,
too many lanes, duplicate labels, subject labels that can't be joined across places, truncated citations.
Treat the numbers as triage; the gold set + a lawyer is the real test.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

B = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower())) - {"and", "the", "of", "a"}


def lint(bpmn: str, sidecar: dict, bundle: dict | None, pass1: str | None, gold: dict | None) -> dict:
    root = ET.parse(bpmn).getroot()
    nodes = {e.get("id"): (e.tag[len(B):], e.get("name") or "", e) for e in root.iter()
             if e.tag in (B + "userTask", B + "startEvent", B + "endEvent", B + "exclusiveGateway")}
    flows = [(f.get("sourceRef"), f.get("targetRef"), f.get("name") or "") for f in root.iter(B + "sequenceFlow")]
    lanes = {l.get("name") for l in root.iter(B + "lane")}
    tasks = {i: v for i, v in nodes.items() if v[0] == "userTask"}
    gateways = [i for i, v in nodes.items() if v[0] == "exclusiveGateway"]
    out_deg = collections.Counter(s for s, _, _ in flows)
    start = next(i for i, v in nodes.items() if v[0] == "startEvent")
    els = {e["id"]: e for e in sidecar.get("elements", [])}
    res: dict = {"findings": [], "metrics": {}}
    F = lambda sev, code, msg: res["findings"].append({"severity": sev, "code": code, "message": msg})
    M = res["metrics"]

    # structure
    M.update(tasks=len(tasks), gateways=len(gateways), lanes=len(lanes), start_fanout=out_deg[start])
    if not gateways:
        F("high", "no-branching", "No gateways: a licensing process always has decisions (class, threshold, compliant?). The chain is just ordered statements.")
    if out_deg[start] > 1:
        F("med", "multiple-roots", f"Start fans out to {out_deg[start]} tasks; the model found no single entry point.")
    if tasks and len(lanes) >= 5 and len(lanes) / len(tasks) > 0.4:
        F("med", "lane-sprawl", f"{len(lanes)} lanes for {len(tasks)} tasks; actor names are not normalised (e.g. 'License inspector' vs 'License inspector and deputies').")
    dup = [k for k, v in collections.Counter(v[1].lower() for v in tasks.values()).items() if v > 1]
    if dup:
        F("low", "duplicate-labels", f"Identical task labels on different nodes: {dup}")
    for kind, name, _ in nodes.values():
        if kind == "endEvent" and re.search(r"\bor\b", name):
            F("med", "merged-outcomes", f"End event '{name}' merges outcomes; split into separate ends behind a gateway.")

    # provenance + source quality (text snippets are embedded in <documentation>)
    snippets: dict[str, str] = {}
    for i, (kind, name, e) in tasks.items():
        d = e.find(B + "documentation")
        text = (d.text or "") if d is not None else ""
        snippets[i] = " ".join(b for b in text.split("\n\n") if b.startswith("[§"))
        body = re.sub(r"^\[[^\]]*\]\s*", "", snippets[i])
        if not body.strip():
            F("high", "unsourced", f"Task '{name}' has no source text.")
        elif len(body.split()) < 8:
            F("high", "near-empty-source", f"Task '{name}' is built from a {len(body.split())}-word chunk ({body.strip()[:40]!r}); the step is almost certainly invented.")

    # scope: provisions outside the dominant chapter
    secs = [(i, s.strip()) for i, e in els.items() for s in str(e.get("sections") or "").split(",") if s.strip()]
    chap = lambda s: re.split(r"[.\-]", s)[0]
    if secs:
        dom, n_dom = collections.Counter(chap(s) for _, s in secs).most_common(1)[0]
        off = sorted({s for _, s in secs if chap(s) != dom})
        M["dominant_chapter"], M["off_chapter_sections"] = dom, off
        if off:
            F("high", "off-topic-sections", f"Sections {off} sit outside chapter {dom}; retrieval matched words, not meaning (e.g. a 'licensed business' carve-out in an unrelated chapter).")
    admin = [e["label"] for e in els.values() if re.search(r"appoint|summon|examine books|certify|pay such refund|ascertain names", e["label"], re.I)]
    if admin:
        F("med", "internal-admin-steps", f"{len(admin)} steps are what the government does internally, not what an applicant must do: {admin[:4]}…")

    # typing: a rate must never be stored as a flat fee
    for e in els.values():
        ctx = f"{e.get('label','')} {e.get('condition','')} {e.get('amount_text','')}"
        if e.get("amount_kind") == "flat_fee" and (re.search(r"per (hundred|\$100)|cents|percent|%", ctx, re.I)
                                                   or "per" in (e.get("amount_unit") or "") or e.get("amount_base")):
            F("high", "rate-as-fee", f"'{e['label']}' is typed flat_fee ({e.get('amount_value')}) but looks like a rate. Exported as a flat fee it would corrupt every comparison.")
        if e.get("fee_usd") and re.search(r"per (hundred|\$100)|cents|percent|%", ctx, re.I):      # v2-era sidecars
            F("high", "rate-as-fee", f"'{e['label']}' stores fee_usd={e['fee_usd']} but the text describes a rate.")
        if e.get("type") == "end" and (e.get("amount_value") or e.get("fee_usd") or e.get("deadline_days")):
            F("med", "end-node-facts", f"End node '{e['label']}' carries facts; outcomes should not.")
    verdicts = collections.Counter(e.get("verdict") or "unverified" for e in els.values() if e.get("type") == "task")
    M["verdicts"] = dict(verdicts)
    if verdicts.get("unsupported"):
        F("high", "unsupported-steps", f"{verdicts['unsupported']} step(s) the verifier says the source text does not state.")
    if verdicts.get("unverified"):
        F("med", "unverified-steps", f"{verdicts['unverified']} step(s) were never verified.")
    M["by_level"] = dict(collections.Counter(l for e in els.values() for l in (e.get("levels") or "").split(",") if l))

    # export
    if bundle:
        subj = bundle["subject"]["label"]
        place = bundle["jurisdiction"]["label"]
        if place.split(",")[0].lower() in subj.lower():
            F("high", "subject-not-joinable", f"Subject '{subj}' contains the jurisdiction; it can never match another city's subject.")
        junk = [i["label"] for i in bundle["items"] if i["key"].startswith("prov:") and re.search(r"§ [A-Z ]+$", i["label"])]
        if junk:
            F("med", "junk-provision-items", f"Provision items created from page-header chunks: {junk}")
        proc = next((i for i in bundle["items"] if i["key"].startswith("proc:")), None)
        rec = next((s["value"] for s in proc["statements"] if s["prop"] == "extraction_record"), "") if proc else ""
        if re.search(r"reviewed-ok\s*$", rec):
            F("med", "review-without-note", "Marked reviewed-ok with no reviewer note; the gate passed with no evidence of a review.")
        actors = [i["label"] for i in bundle["items"] if i["key"].startswith("actor:")]
        near = [(a, b) for k, a in enumerate(actors) for b in actors[k + 1:] if len(toks(a) & toks(b)) / max(1, len(toks(a) | toks(b))) >= 0.5]
        if near:
            F("low", "near-duplicate-actors", f"Likely the same role: {near}")
    if pass1 is not None:
        trunc = len(re.findall(r"…\"", pass1))
        if trunc:
            F("low", "truncated-citations", f"{trunc} citation(s) end in '…' because the header was cut at 120 chars.")
        if not re.search(r"\tP5\t", pass1):
            F("info", "pass1-incomplete", "pass 1 has no instance-of statements: class/jurisdiction QIDs are not in qid_map yet.")

    # gold fact recall — CRUDE upper bound: a regex hit can come from an unrelated step (seen: "$50 to $500 per offense" matching the $50 flat fee)
    if gold:
        blob = " ".join(json.dumps(e) for e in els.values()).lower() + " " + (json.dumps(bundle).lower() if bundle else "")
        hit = {f["id"]: all(re.search(p, blob, re.I) for p in f["patterns"]) for f in gold["applicant_facts"]}
        M["gold_fact_recall"] = f"{sum(hit.values())}/{len(hit)}"
        M["gold_missing"] = [k for k, v in hit.items() if not v]
    return res


def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpmn", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--bundle")
    ap.add_argument("--pass1")
    ap.add_argument("--gold")
    a = ap.parse_args()
    r = lint(a.bpmn, json.loads(Path(a.sidecar).read_text(encoding="utf-8")),
             json.loads(Path(a.bundle).read_text(encoding="utf-8")) if a.bundle else None,
             Path(a.pass1).read_text(encoding="utf-8") if a.pass1 else None,
             json.loads(Path(a.gold).read_text(encoding="utf-8")) if a.gold else None)
    print(json.dumps(r["metrics"], indent=2))
    order = {"high": 0, "med": 1, "low": 2, "info": 3}
    for f in sorted(r["findings"], key=lambda f: order[f["severity"]]):
        print(f"[{f['severity'].upper():4}] {f['code']}: {f['message']}")


if __name__ == "__main__":
    main()
