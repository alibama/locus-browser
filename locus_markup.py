"""
locus_markup — the upstream markup that coordinates process models, plus the pipeline that produces it.

Instead of asking one model to turn raw sections straight into a diagram, each section is marked up in
several cheap, constrained passes whose results are stored and reusable (see docs/markup-schema.md):

    0 quality     (no LLM)  word count, heading/page-artifact, truncated?
    1 triage      what *kind* of text is this (role), who is it for (audience), which controlled subjects,
                  which legal scheme (regime), plain-English summary, cross-references
    1b query      what process is the researcher actually asking about (goal, audience, subjects)
    2 relevance   is this section part of THAT process for THAT audience?  (query-specific; stores a reason)
    3 extract     typed facts, routed by role: rules / rate tables / definitions; every rule carries a verbatim
                  evidence quote that is checked against the source text
    4 regime      group the relevant sections into phases (scope → … → penalty), list what is excluded and
                  what a complete process would normally have but we did not find
    5 compile     build the graph from the phase-ordered rules; rate tables become decision tables
    6 verify      a (preferably different) model checks each cited rule against its source and must quote it

Every LLM call is logged; every result is stored keyed by content hash + model + schema version.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

import locus_core as C
from locus_core import DIM_LIST, clean, grounded_num, label, quote_in, sec_of
from locus_llm import LLMUnavailable, call_llm

# ───────────────────────────── vocabularies ─────────────────────────────
ROLES = ["definition", "applicability", "duty", "procedure", "rate_schedule", "exemption", "penalty",
         "admin_power", "authority", "heading_or_artifact", "other"]
AUDIENCES = ["regulated_person", "official", "court", "general"]
PHASES = ["scope", "preconditions", "application", "determination", "issuance", "term_and_renewal",
          "ongoing_duties", "violation_and_penalty", "appeal", "out_of_scope"]
RELATIONS = ["exception_to", "defined_in", "penalty_in", "procedure_in", "rate_in", "authorizes", "other"]
RULE_TYPES = ["duty", "prohibition", "permission", "procedure_step", "exemption", "threshold", "classification",
              "penalty", "power"]
AMOUNT_KINDS = ["none", "flat_fee", "percent_of_base", "per_unit", "tiered"]
PENALTY_KINDS = ["none", "fine", "misdemeanor", "infraction", "interest", "revocation", "other"]
RENEWALS = ["annual", "biennial", "one-time", "none", "unspecified"]
OPS = ["", "<", "<=", ">", ">=", "="]
VERDICTS = ["supported", "partial", "unsupported"]
RELEVANCE = ["yes", "partial", "no"]
ROLE_TO_PHASE = {"definition": "scope", "applicability": "scope", "exemption": "scope", "authority": "scope",
                 "duty": "preconditions", "procedure": "application", "rate_schedule": "determination",
                 "penalty": "violation_and_penalty", "admin_power": "out_of_scope"}
DEONTIC = {"duty": "obligation", "procedure_step": "obligation", "prohibition": "prohibition",
           "permission": "permission", "power": "power"}

# ───────────────────────────── JSON schemas (used as Ollama `format`) ─────────────────────────────
_S = {"type": "string"}


def _enum(vals):
    return {"type": "string", "enum": list(vals)}


def _arr(item, n=None):
    a = {"type": "array", "items": item}
    if n:
        a["maxItems"] = n
    return a


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": list(required or props)}


TRIAGE_SCHEMA = _obj({
    "role": _enum(ROLES), "audience": _enum(AUDIENCES), "subjects": _arr(_S, 4), "regime": _S, "summary": _S,
    "self_contained": {"type": "boolean"},
    "refs": _arr(_obj({"target": _S, "relation": _enum(RELATIONS)}), 8)})
QUERY_SCHEMA = _obj({"goal": _S, "audience": _enum(AUDIENCES), "subjects": _arr(_S, 4), "keywords": _arr(_S, 6)})
RELEVANCE_SCHEMA = _obj({"verdict": _enum(RELEVANCE), "reason": _S, "phase": _enum(PHASES)})

_RULE = {
    "rule_type": _enum(RULE_TYPES), "actor": _S, "action": _S, "condition": _S, "applies_if": _arr(_S, 4),
    "deadline_text": _S, "deadline_days": _S, "deadline_fixed": _S, "deadline_anchor": _S,
    "amount_text": _S, "amount_kind": _enum(AMOUNT_KINDS), "amount_value": _S, "amount_unit": _S, "amount_base": _S,
    "threshold_variable": _S, "threshold_op": _enum(OPS), "threshold_value": _S, "threshold_unit": _S,
    "penalty_text": _S, "penalty_kind": _enum(PENALTY_KINDS), "penalty_max_value": _S, "penalty_unit": _S,
    "renewal": _enum(RENEWALS), "evidence": _S}
RULES_SCHEMA = _obj({"rules": _arr(_obj(_RULE), 6)})
_ROW = {"class_label": _S, "covers": _S, "amount_kind": _enum(AMOUNT_KINDS), "amount_value": _S, "amount_unit": _S,
        "amount_base": _S, "min_receipts": _S, "max_receipts": _S, "note": _S, "evidence": _S}
RATES_SCHEMA = _obj({"table_title": _S, "rows": _arr(_obj(_ROW), 16)})
DEFS_SCHEMA = _obj({"terms": _arr(_obj({"term": _S, "meaning": _S}), 8)})
REGIME_SCHEMA = _obj({
    "title": _S, "subject": _S,
    "phases": _arr(_obj({"phase": _enum(PHASES), "codes": _arr(_S), "note": _S})),
    "excluded": _arr(_obj({"code": _S, "reason": _S})),
    "missing": _arr(_obj({"item": _S, "why": _S}), 8)})
COMPILE_SCHEMA = _obj({
    "title": _S,
    "nodes": _arr(_obj({"id": _S, "type": _enum(["task", "gateway", "end"]), "actor": _S, "label": _S,
                        "sources": _arr(_S), "decision": _S}), 24),
    "edges": _arr(_obj({"from": _S, "to": _S, "label": _S}, ["from", "to"]))})
VERIFY_SCHEMA = _obj({"verdict": _enum(VERDICTS), "issues": _arr(_S, 4), "quote": _S})

# ───────────────────────────── prompts ─────────────────────────────
TRIAGE_SYSTEM = (
    "You label ONE section of US law for a legal-process mapping project. Use only the text given.\n"
    "role: definition | applicability (who/what is covered, classes, scope) | duty (requires or prohibits something "
    "of a regulated person) | procedure (steps for applying, filing, appealing) | rate_schedule (fees, taxes, rates, "
    "tables) | exemption (exceptions, or a threshold below which the rule does not apply) | penalty (sanctions for "
    "violating) | admin_power (powers and duties of officials that ask nothing of the regulated person: appointing staff, "
    "audits, how refunds are paid) | authority (higher-level law that authorises or limits what a locality may do) | "
    "heading_or_artifact (heading, page header, 'reserved', a fragment with no rule) | other.\n"
    "audience: regulated_person | official | court | general.\n"
    "subjects: up to 4 slugs from this list: {subjects}. If none fits use 'new:<short-kebab-slug>'.\n"
    "regime: a 2-5 word name for the legal scheme this section belongs to (e.g. 'business license tax').\n"
    "summary: one plain-English sentence a non-lawyer can follow.\n"
    "self_contained: false if the text is cut off or only makes sense with neighbouring sections.\n"
    "refs: other sections it points to, with how (exception_to, defined_in, penalty_in, procedure_in, rate_in, authorizes, other).")
QUERY_SYSTEM = (
    "A researcher typed a topic (it may be a word stem, e.g. 'dog licen'). Say what process a member of the public "
    "would go through. goal: one sentence starting with a verb. audience: usually regulated_person. subjects: up to 4 "
    "slugs from this list: {subjects} (or 'new:<slug>'). keywords: 3-6 plain search words.")
RELEVANCE_SYSTEM = (
    "Decide whether a section belongs in a step-by-step process map for the GOAL below, from the point of view of the "
    "AUDIENCE. yes = directly part of what they must do, pay, file or face. partial = applies only to a sub-group, or "
    "is needed context (definition, threshold, authority). no = a different subject, a different class of business or "
    "activity than the goal, or only about what officials do internally. If a section belongs to the same legal scheme as the "
    "topic (same chapter, same tax or license) and you are unsure between partial and no, choose partial. Give a one-sentence "
    "reason and the phase this section belongs to: " + ", ".join(PHASES) + " (out_of_scope if verdict is no).")
RULES_SYSTEM = (
    "Extract the rules stated in ONE section of US law. Use only the text; if something is not stated return an empty "
    "string and never guess. One entry per distinct rule (max 6). rule_type: duty, prohibition, permission, "
    "procedure_step, exemption, threshold (a number that decides whether/how a rule applies), classification (which "
    "class something falls in), penalty, power (an official may act). actor: short consistent role name "
    "('Business owner', 'Commissioner of the revenue'). action: imperative, at most 12 words. applies_if: short phrases. "
    "Numbers (deadline_days, amount_value, threshold_value, penalty_max_value) are digits only, copied from the text. "
    "amount_kind: flat_fee ONLY for a fixed dollar amount; percent_of_base for a rate such as '36 cents per $100 of "
    "gross receipts' (amount_value 0.36, amount_unit usd_per_100, amount_base 'gross receipts'); per_unit; tiered. "
    "Never put a rate in a flat fee. deadline_fixed: a calendar date like 'March 1'. "
    "evidence: a verbatim quote of at most 25 words from the text that supports this rule.")
RATES_SYSTEM = (
    "The section contains a fee or tax schedule. Extract each row. class_label: the class/category name as written. "
    "covers: what activities it covers. amount_kind: flat_fee | percent_of_base (a rate per $100 or %) | per_unit | "
    "tiered. amount_value digits copied from the text; amount_base what the rate applies to ('gross receipts'). "
    "min_receipts / max_receipts: the gross-receipts range for the row if given, digits only. Use only the text; "
    "empty string if not stated. evidence: a verbatim quote (max 25 words).")
DEFS_SYSTEM = "List the terms this section defines, each with a one-sentence plain-English meaning. Use only the text."
REGIME_SYSTEM = (
    "You organise sections of law into ONE process for the goal below. Assign every section code to a phase: "
    + ", ".join(PHASES) + ". Put sections that only apply to a different class, or are about officials' internal work, "
    "in 'excluded' with a reason (a code may appear once). title: a short process name WITHOUT the place name. "
    "subject: the best slug from this list: {subjects}. missing: things a complete process of this kind normally "
    "specifies (who to apply to, deadline, fee, renewal, penalty for not complying, prerequisites such as zoning) that "
    "NONE of the sections state. Use only the sections given.")
COMPILE_SYSTEM = (
    "Assemble the given rules into ONE process model a lawyer can check. Use ONLY the rules listed; never add steps, "
    "amounts or deadlines. Node types: task (an actor does something), gateway (a question; label it as a question and "
    "label every outgoing edge with the answer — use gateways for applicability, thresholds, classes, 'compliant?'), "
    "end (an outcome such as 'License issued'). Each task lists the rule codes it comes from in 'sources' (e.g. c3.1). "
    "A task that determines a fee may name a decision table in 'decision' (e.g. D1) instead of repeating its rows. "
    "Follow the phase order given and the logical sequence of deadlines and conditions. Use one consistent actor name "
    "per role. Rules marked [state] or [federal] are higher-level law constraining the local rules. "
    "Do not connect steps unless the text gives a reason. At most 18 nodes.")
VERIFY_SYSTEM = (
    "You check whether a CLAIM about a section of law is supported by the TEXT. supported = the text states it; "
    "partial = the text supports part of it or states it differently; unsupported = the text does not say it. "
    "List issues briefly. quote: copy verbatim (max 25 words) the part of the text that supports it, or empty.")


# ───────────────────────────── normalisers (never trust model output) ─────────────────────────────
def _e(v, allowed, default):
    v = clean(v).strip().lower()
    return v if v in allowed else default


def _s(v, n=300):
    return re.sub(r"\s+", " ", clean(v)).strip()[:n]


def quality_flags(text: str, header: str = "") -> dict:
    """Deterministic pre-check. A section is an artifact (page header, bare cross-reference, 'Reserved') when it has
    fewer than 3 real words once section numbers and symbols are ignored. Word COUNT alone is a bad test:
    'The annual dog license fee is $8.' is short and perfectly substantive."""
    t = clean(text).strip()
    words = len(t.split())
    real = [w for w in re.findall(r"[A-Za-z]{3,}", t) if w.lower() not in {"sec", "section", "sections", "subsection"}]
    artifact = len(real) < 3
    return {"words": words, "real_words": len(real), "artifact": artifact,
            "truncated": (bool(t) and not re.search(r"[.;:)\"”]\s*$", t)) or t.endswith(":"),
            "reason": "fewer than 3 real words (page header, bare reference or 'reserved')" if artifact else ""}


def norm_triage(d: dict, q: dict) -> dict:
    subs = []
    for s in d.get("subjects") or []:
        s = _s(s, 60).lower()
        s = ("new:" + C.slug(s[4:])) if s.startswith("new:") else C.slug(s)
        if s and s not in subs:
            subs.append(s)
    refs = [{"target": _s(r.get("target"), 40), "relation": _e(r.get("relation"), RELATIONS, "other")}
            for r in (d.get("refs") or []) if isinstance(r, dict) and _s(r.get("target"), 40)]
    return {"role": _e(d.get("role"), ROLES, "other"), "audience": _e(d.get("audience"), AUDIENCES, "general"),
            "subjects": subs[:4], "regime": _s(d.get("regime"), 60), "summary": _s(d.get("summary"), 300),
            "self_contained": bool(d.get("self_contained", True)), "refs": refs[:8], "quality": q}


def norm_rules(d: dict, text: str) -> list[dict]:
    out = []
    for r in (d.get("rules") or [])[:6]:
        if not isinstance(r, dict):
            continue
        ev = _s(r.get("evidence"), 400)
        amt, thr, pen, dl = _s(r.get("amount_text")), _s(r.get("threshold_value")), _s(r.get("penalty_text")), _s(r.get("deadline_text"))
        o = {
            "rule_type": _e(r.get("rule_type"), RULE_TYPES, "duty"), "actor": _s(r.get("actor"), 80),
            "action": _s(r.get("action"), 160), "condition": _s(r.get("condition")),
            "applies_if": [_s(x, 160) for x in (r.get("applies_if") or []) if _s(x, 160)][:4],
            "deadline_text": dl, "deadline_days": grounded_num(r.get("deadline_days"), f"{dl} {text}"),
            "deadline_fixed": _s(r.get("deadline_fixed"), 40), "deadline_anchor": _s(r.get("deadline_anchor"), 80),
            "amount_text": amt, "amount_kind": _e(r.get("amount_kind"), AMOUNT_KINDS, "none"),
            "amount_value": grounded_num(r.get("amount_value"), f"{amt} {text}"),
            "amount_unit": _s(r.get("amount_unit"), 30), "amount_base": _s(r.get("amount_base"), 60),
            "threshold_variable": _s(r.get("threshold_variable"), 60), "threshold_op": _e(r.get("threshold_op"), OPS, ""),
            "threshold_value": grounded_num(r.get("threshold_value"), text), "threshold_unit": _s(r.get("threshold_unit"), 30),
            "penalty_text": pen, "penalty_kind": _e(r.get("penalty_kind"), PENALTY_KINDS, "none"),
            "penalty_max_value": grounded_num(r.get("penalty_max_value"), f"{pen} {text}"),
            "penalty_unit": _s(r.get("penalty_unit"), 30), "renewal": _e(r.get("renewal"), RENEWALS, "unspecified"),
            "evidence": ev, "evidence_ok": quote_in(ev, text)}
        if o["amount_kind"] != "none" and not o["amount_value"]:
            o["amount_kind"] = "none" if o["amount_kind"] == "flat_fee" else o["amount_kind"]   # unverifiable flat fee -> not a fee
        if o["action"] or o["threshold_value"]:
            out.append(o)
    return out


def norm_rows(d: dict, text: str) -> list[dict]:
    out = []
    for r in (d.get("rows") or [])[:16]:
        if not isinstance(r, dict):
            continue
        ev = _s(r.get("evidence"), 400)
        row = {"class_label": _s(r.get("class_label"), 120), "covers": _s(r.get("covers"), 300),
               "amount_kind": _e(r.get("amount_kind"), AMOUNT_KINDS, "none"),
               "amount_value": grounded_num(r.get("amount_value"), text), "amount_unit": _s(r.get("amount_unit"), 30),
               "amount_base": _s(r.get("amount_base"), 60), "min_receipts": grounded_num(r.get("min_receipts"), text),
               "max_receipts": grounded_num(r.get("max_receipts"), text), "note": _s(r.get("note")),
               "evidence": ev, "evidence_ok": quote_in(ev, text)}
        if row["class_label"] or row["amount_value"]:
            out.append(row)
    return out


def claim_of(rule: dict) -> dict:
    keys = ("rule_type", "actor", "action", "condition", "applies_if", "deadline_text", "deadline_days", "deadline_fixed",
            "amount_text", "amount_kind", "amount_value", "amount_unit", "amount_base", "threshold_variable",
            "threshold_op", "threshold_value", "penalty_text", "penalty_max_value", "renewal")
    return {k: rule[k] for k in keys if rule.get(k) not in ("", [], None, "none", "unspecified")}


# ───────────────────────────── graph normalisation (shared by BPMN + export) ─────────────────────────────
def normalize_graph(g: dict, chunk_codes: set[str], rule_codes: set[str] | None = None, decision_ids: set[str] | None = None):
    """Sanitise an LLM graph. 'sources' are split into rule codes (c3.1) and chunk codes (c3)."""
    rule_codes, decision_ids = rule_codes or set(), decision_ids or set()
    used_decisions: set[str] = set()
    nodes: dict[str, dict] = {}
    order: list[str] = []
    for n in g.get("nodes", []) or []:
        nid = clean(n.get("id")).strip()
        if not nid or nid in nodes:
            continue
        t = n.get("type") if n.get("type") in ("task", "gateway", "end") else "task"
        srcs, rules = [], []
        for s in n.get("sources", []) or []:
            code = clean(s).strip()
            if code in rule_codes and code not in rules:
                rules.append(code)
            base = code.split(".")[0]
            if base in chunk_codes and base not in srcs:
                srcs.append(base)
        dec = clean(n.get("decision")).strip()
        # a decision table hangs off ONE task: gateways/outcomes can't own one, and only the first claimant keeps it
        # (a model that stamps 'D1' on every node is not saying anything)
        dec = dec if (dec in decision_ids and t == "task" and dec not in used_decisions) else ""
        if dec:
            used_decisions.add(dec)
        nodes[nid] = {"id": nid, "type": t, "actor": clean(n.get("actor")).strip(),
                      "label": clean(n.get("label")).strip() or nid, "sources": srcs, "rules": rules, "decision": dec}
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


def node_facts(nd: dict, rules_by_code: dict) -> dict:
    """First non-empty typed fact across the rules a node cites (end nodes never carry facts)."""
    out: dict = {}
    if nd.get("type") == "end":
        return out
    for c in nd.get("rules", []):
        r = rules_by_code.get(c)
        if not r:
            continue
        mod = DEONTIC.get(r.get("rule_type", ""))
        cand = {"modality": mod, "condition": "; ".join(filter(None, [r.get("condition"), *r.get("applies_if", [])])),
                "deadline_days": r.get("deadline_days"), "deadline_fixed": r.get("deadline_fixed"),
                "amount_kind": r.get("amount_kind") if r.get("amount_kind") != "none" else "",
                "amount_value": r.get("amount_value"), "amount_unit": r.get("amount_unit"),
                "amount_base": r.get("amount_base"), "amount_text": r.get("amount_text"),
                "threshold": " ".join(filter(None, [r.get("threshold_variable"), r.get("threshold_op"), r.get("threshold_value"), r.get("threshold_unit")])),
                "penalty_text": r.get("penalty_text"), "penalty_max_value": r.get("penalty_max_value"),
                "penalty_unit": r.get("penalty_unit"), "renewal": r.get("renewal") if r.get("renewal") != "unspecified" else ""}
        for k, v in cand.items():
            if v and k not in out:
                out[k] = v
    return out


# ───────────────────────────── jobs, progress, parallel plumbing ─────────────────────────────
class Progress:
    """Thread-safe progress board; worker threads write, the Streamlit main thread reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}

    def set(self, key: str, **kw):
        with self.lock:
            self.state.setdefault(key, {}).update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            return {k: dict(v) for k, v in self.state.items()}


@dataclass
class Job:
    state: str
    place: str
    query: str
    chunks: pd.DataFrame
    model: str
    host: str
    verifier: str = ""
    audience: str = "regulated_person"
    workers: int = 1
    goal: str = ""             # optional human-written goal; skips the model's guess
    timeout: int = 1800        # seconds per model call
    abort: threading.Event = field(default_factory=threading.Event)

    @property
    def pid(self) -> str:
        return f"{self.place} · {self.model}"

    @property
    def query_key(self) -> str:
        return hashlib.sha1((self.query.strip().lower() + "|" + self.goal.strip().lower()).encode()).hexdigest()[:12]

    @property
    def ver(self) -> str:
        return self.verifier or self.model


def _map(job: Job, prog: Progress, stage: str, items: list, fn) -> list[str]:
    """Run fn(item) over items with job.workers threads; update progress; stop on server loss."""
    errs: list[str] = []
    total = len(items)
    prog.set(job.pid, stage=stage, done=0, total=total)
    if not total:
        return errs

    def wrap(it):
        if job.abort.is_set():
            return "aborted"
        try:
            fn(it)
            return ""
        except LLMUnavailable:
            job.abort.set()
            raise
        except Exception as e:  # noqa: BLE001
            return f"{stage}: {e}"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, job.workers)) as ex:
        futs = [ex.submit(wrap, it) for it in items]
        try:
            for f in as_completed(futs):
                r = f.result()
                done += 1
                prog.set(job.pid, done=done)
                if r and r != "aborted":
                    errs.append(r)
        except LLMUnavailable:
            for f in futs:
                f.cancel()
            raise
    return errs


def _chunk_text(r) -> str:
    return clean(r["content"])[:3500]


def _hdr(job: Job, r) -> str:
    lvl = {"local": "local ordinance", "state": "state law", "federal": "federal law"}.get(r["level"], r["level"])
    where = r["place"] if r["level"] == "local" else r["state"].upper()
    return f"Level: {lvl}\nJurisdiction: {where}\nSection: {label(r['header'], 160)}"


def _subject_list() -> str:
    return ", ".join(sorted(C.subjects(include_proposed=False)))


# ───────────────────────────── stages ─────────────────────────────
def pipeline_status(job: Job) -> dict:
    """Counts of what is already stored for this job (no LLM calls)."""
    keys = job.chunks["ckey"].tolist()
    tri = C.markup_get(keys, "triage", job.model)
    rel = C.relevance_get(keys, job.query_key, job.model)
    ex = C.markup_get(keys, "extract", job.model)
    eligible = [k for k in keys if rel.get(k, {}).get("verdict") in ("yes", "partial")]
    pkey = process_key(job)
    rec = C.get_process(pkey)
    return {"chunks": len(keys), "triage": len(tri), "relevance": len(rel), "relevant": len(eligible),
            "extract": sum(1 for k in eligible if k in ex), "query": C.query_get(job.query_key, job.model) is not None,
            "regime": C.regime_get(regime_key(job)) is not None, "process": rec is not None,
            "verified": len(rec.get("verdicts", {})) if rec else 0, "pkey": pkey, "triage_data": tri,
            "relevance_data": rel, "extract_data": ex}


DEFAULT_SECONDS = {"triage": 60, "relevance": 40, "extract:rules": 120, "extract:rates": 120, "query": 30, "regime": 90,
                   "compile": 120, "verify": 40}


def estimate(job: Job) -> dict:
    """Rough time left, from this machine's own earlier call timings for this model (defaults otherwise)."""
    st = pipeline_status(job)
    stats = C.llm_stats(job.model)
    vstats = C.llm_stats(job.ver) if job.ver != job.model else stats
    sec = lambda stage, s=stats: s.get(stage, (DEFAULT_SECONDS[stage], 0))[0]
    n = st["chunks"]
    relevant = st["relevant"] if st["relevance"] >= n else max(1, round(0.4 * n))
    calls = {"triage": max(0, n - st["triage"]), "relevance": max(0, n - st["relevance"]),
             "extract": max(0, relevant - st["extract"]), "query": 0 if st["query"] else 1,
             "regime": 0 if st["regime"] else 1, "compile": 0 if st["process"] else 1,
             "verify": 0 if st["verified"] else min(10, max(3, relevant))}
    ex = (sec("extract:rules") + sec("extract:rates")) / 2
    secs = (calls["triage"] * sec("triage") + calls["relevance"] * sec("relevance") + calls["extract"] * ex
            + calls["query"] * sec("query") + calls["regime"] * sec("regime") + calls["compile"] * sec("compile")
            + calls["verify"] * sec("verify", vstats))
    seen = sum(n_ for _, n_ in stats.values())
    return {"minutes": round(secs / 60), "calls": sum(calls.values()), "basis": f"{seen} earlier calls" if seen else "default guesses"}


def regime_key(job: Job) -> str:
    return hashlib.sha1("|".join([job.state, job.place, job.query.strip().lower(), job.model, C.SCHEMA_VER,
                                  *sorted(job.chunks["ckey"])]).encode()).hexdigest()[:20]


def process_key(job: Job) -> str:
    return regime_key(job)[::-1]   # same inputs, distinct namespace


def stage_triage(job: Job, prog: Progress) -> list[str]:
    keys = job.chunks["ckey"].tolist()
    have = C.markup_get(keys, "triage", job.model)
    todo = [r for _, r in job.chunks.iterrows() if r["ckey"] not in have]
    sysmsg = TRIAGE_SYSTEM.format(subjects=_subject_list())

    def one(r):
        q = quality_flags(r["content"], r["header"])
        if q["artifact"]:
            C.markup_put(r["ckey"], "triage", job.model, {
                "role": "heading_or_artifact", "audience": "general", "subjects": [], "regime": "", "summary": "",
                "self_contained": False, "refs": [], "quality": q, "skipped_llm": True})
            return
        d = call_llm(job.model, job.host, sysmsg, f"{_hdr(job, r)}\n\nText:\n{_chunk_text(r)}", TRIAGE_SCHEMA,
                     stage="triage", ctx=r["ckey"], timeout=job.timeout)
        t = norm_triage(d, q)
        for s in t["subjects"]:
            if s.startswith("new:"):
                C.subject_add(s[4:], s[4:].replace("-", " ").capitalize(), "proposed")
        t["subjects"] = [s[4:] if s.startswith("new:") else s for s in t["subjects"]]
        C.markup_put(r["ckey"], "triage", job.model, t)

    return _map(job, prog, "triage", todo, one)


def goal_ok(g: str) -> bool:
    """A usable goal is a real sentence ('Obtain a business license'), not a stray word like 'search'."""
    return len(g.split()) >= 4 and len(g) >= 20


def default_goal(query: str) -> str:
    return f"Find out what a person must do, pay, file or face under the local rules about {query.strip()}"


def stage_query(job: Job, prog: Progress) -> list[str]:
    if C.query_get(job.query_key, job.model):
        return []
    prog.set(job.pid, stage="query", done=0, total=1)
    if job.goal.strip():           # a human-written goal beats a guess: no model call
        C.query_put(job.query_key, job.model, {"goal": _s(job.goal, 300), "audience": job.audience, "subjects": [],
                                               "keywords": [], "by": "user"})
        prog.set(job.pid, done=1)
        return []
    d = call_llm(job.model, job.host, QUERY_SYSTEM.format(subjects=_subject_list()), f"Topic typed: {job.query}\n"
                 f"Jurisdiction: {job.place}, {job.state.upper()}", QUERY_SCHEMA, stage="query", ctx=job.query_key,
                 timeout=job.timeout)
    subs = [C.slug(s[4:]) if s.startswith("new:") else C.slug(s) for s in (d.get("subjects") or [])][:4]
    g, errs = _s(d.get("goal"), 200), []
    by = "llm"
    if not goal_ok(g):
        errs.append(f"query: the model's goal ({g!r}) was unusable, so a default goal is used. Type your own goal above to override it.")
        g, by = default_goal(job.query), "fallback"
    C.query_put(job.query_key, job.model, {
        "goal": g, "audience": _e(d.get("audience"), AUDIENCES, job.audience), "subjects": subs,
        "keywords": [_s(k, 30) for k in (d.get("keywords") or [])][:6], "by": by})
    prog.set(job.pid, done=1)
    return errs


def stage_relevance(job: Job, prog: Progress) -> list[str]:
    keys = job.chunks["ckey"].tolist()
    have = C.relevance_get(keys, job.query_key, job.model)
    tri = C.markup_get(keys, "triage", job.model)
    qi = C.query_get(job.query_key, job.model) or {"goal": job.query, "audience": job.audience}
    todo = [r for _, r in job.chunks.iterrows() if r["ckey"] not in have]

    def one(r):
        t = tri.get(r["ckey"])
        if t is None or t["role"] == "heading_or_artifact":
            C.relevance_put(r["ckey"], job.query_key, job.model,
                            {"verdict": "no", "reason": "heading or page artifact (not enough text)", "phase": "out_of_scope", "by": "rule"})
            return
        user = (f"GOAL: {qi['goal']}\nTOPIC the researcher typed: {job.query}\nAUDIENCE: {qi['audience']}\n\n{_hdr(job, r)}\nTriage: role={t['role']}, "
                f"audience={t['audience']}, regime={t['regime']}\nSummary: {t['summary']}\n\nText:\n{clean(r['content'])[:2500]}")
        d = call_llm(job.model, job.host, RELEVANCE_SYSTEM, user, RELEVANCE_SCHEMA, stage="relevance", ctx=r["ckey"], timeout=job.timeout)
        C.relevance_put(r["ckey"], job.query_key, job.model,
                        {"verdict": _e(d.get("verdict"), RELEVANCE, "partial"), "reason": _s(d.get("reason"), 240),
                         "phase": _e(d.get("phase"), PHASES, ROLE_TO_PHASE.get(t["role"], "scope")), "by": "llm"})

    return _map(job, prog, "relevance", todo, one)


def stage_extract(job: Job, prog: Progress) -> list[str]:
    keys = job.chunks["ckey"].tolist()
    tri = C.markup_get(keys, "triage", job.model)
    rel = C.relevance_get(keys, job.query_key, job.model)
    have = C.markup_get(keys, "extract", job.model)
    todo = [r for _, r in job.chunks.iterrows()
            if rel.get(r["ckey"], {}).get("verdict") in ("yes", "partial") and r["ckey"] not in have and r["ckey"] in tri]

    def one(r):
        t, text = tri[r["ckey"]], clean(r["content"])
        body = f"{_hdr(job, r)}\nTriage role: {t['role']}\n\nText:\n{_chunk_text(r)}"
        if t["role"] == "rate_schedule":
            d = call_llm(job.model, job.host, RATES_SYSTEM, body, RATES_SCHEMA, stage="extract:rates", ctx=r["ckey"], timeout=job.timeout)
            out = {"kind": "rates", "table_title": _s(d.get("table_title"), 160), "rows": norm_rows(d, text), "rules": []}
        elif t["role"] == "definition":
            d = call_llm(job.model, job.host, DEFS_SYSTEM, body, DEFS_SCHEMA, stage="extract:defs", ctx=r["ckey"], timeout=job.timeout)
            out = {"kind": "defs", "terms": [{"term": _s(x.get("term"), 80), "meaning": _s(x.get("meaning"), 240)}
                                             for x in (d.get("terms") or []) if isinstance(x, dict)], "rules": []}
        else:
            d = call_llm(job.model, job.host, RULES_SYSTEM, body, RULES_SCHEMA, stage="extract:rules", ctx=r["ckey"], timeout=job.timeout)
            out = {"kind": "rules", "rules": norm_rules(d, text)}
        C.markup_put(r["ckey"], "extract", job.model, out)

    return _map(job, prog, "extract", todo, one)


def _fallback_regime(job: Job, relevant: pd.DataFrame, tri: dict, rel: dict) -> dict:
    ph: dict[str, list[str]] = {}
    for _, r in relevant.iterrows():
        p = rel[r["ckey"]].get("phase") or ROLE_TO_PHASE.get(tri[r["ckey"]]["role"], "scope")
        ph.setdefault(p, []).append(r["code"])
    return {"title": job.query.strip().title(), "subject": "", "phases": [{"phase": p, "codes": c, "note": ""} for p, c in ph.items()],
            "excluded": [], "missing": [], "by": "fallback"}


def stage_regime(job: Job, prog: Progress) -> list[str]:
    rk = regime_key(job)
    if C.regime_get(rk):
        return []
    keys = job.chunks["ckey"].tolist()
    tri, rel = C.markup_get(keys, "triage", job.model), C.relevance_get(keys, job.query_key, job.model)
    qi = C.query_get(job.query_key, job.model) or {"goal": job.query}
    ok = job.chunks[[rel.get(k, {}).get("verdict") in ("yes", "partial") for k in job.chunks["ckey"]]]
    if ok.empty:
        return ["No relevant sections to organise."]
    prog.set(job.pid, stage="regime", done=0, total=1)
    lines = []
    for _, r in ok.iterrows():
        t, rl = tri[r["ckey"]], rel[r["ckey"]]
        refs = ", ".join(f"{x['relation']}:{x['target']}" for x in t.get("refs", [])) or "-"
        lines.append(f"{r['code']} | [{r['level']}] §{sec_of(r) or '?'} | role={t['role']} | relevance={rl['verdict']} | "
                     f"{t['summary']} | refs: {refs}")
    # sections judged irrelevant are listed so the model can see what was set aside
    for _, r in job.chunks.iterrows():
        if r["ckey"] not in set(ok["ckey"]) and r["ckey"] in rel:
            lines.append(f"{r['code']} | [{r['level']}] §{sec_of(r) or '?'} | EXCLUDED earlier: {rel[r['ckey']]['reason']}")
    try:
        d = call_llm(job.model, job.host, REGIME_SYSTEM.format(subjects=_subject_list()),
                     f"GOAL: {qi['goal']}\nJurisdiction: {job.place}, {job.state.upper()}\n\nSections:\n" + "\n".join(lines),
                     REGIME_SCHEMA, stage="regime", ctx=job.query_key, timeout=job.timeout)
    except LLMUnavailable:
        raise
    except Exception as e:  # noqa: BLE001  — keep going with a deterministic outline
        C.regime_put(rk, job.state, job.place, job.query, job.model, _fallback_regime(job, ok, tri, rel))
        prog.set(job.pid, done=1)
        return [f"regime: model output unusable ({e}); used a role-based outline instead"]
    valid = set(ok["code"])
    seen: set[str] = set()
    phases = []
    for p in d.get("phases") or []:
        codes = [c for c in (clean(x).strip() for x in p.get("codes", [])) if c in valid and c not in seen]
        seen.update(codes)
        if codes:
            phases.append({"phase": _e(p.get("phase"), PHASES, "scope"), "codes": codes, "note": _s(p.get("note"))})
    leftover = [c for c in ok["code"] if c not in seen]
    if leftover:
        phases.append({"phase": "scope", "codes": leftover, "note": "unassigned by model"})
    C.regime_put(rk, job.state, job.place, job.query, job.model, {
        "title": _s(d.get("title"), 120) or job.query.title(), "subject": C.slug(_s(d.get("subject"), 60)) if d.get("subject") else "",
        "phases": phases, "excluded": [{"code": _s(x.get("code"), 12), "reason": _s(x.get("reason"), 200)}
                                       for x in d.get("excluded") or [] if isinstance(x, dict)][:12],
        "missing": [{"item": _s(x.get("item"), 120), "why": _s(x.get("why"), 200)}
                    for x in d.get("missing") or [] if isinstance(x, dict)][:8], "by": "llm"})
    prog.set(job.pid, done=1)
    return []


def _rule_line(code: str, level: str, sec: str, r: dict) -> str:
    bits = [code, f"[{level}] §{sec or '?'}", f"{r['rule_type']}", f"actor: {r['actor'] or '?'}", f"action: {r['action'] or '-'}"]
    if r["condition"] or r["applies_if"]:
        bits.append("if: " + "; ".join(filter(None, [r["condition"], *r["applies_if"]])))
    if r["threshold_value"]:
        bits.append(f"threshold: {r['threshold_variable']} {r['threshold_op']} {r['threshold_value']} {r['threshold_unit']}".strip())
    if r["deadline_text"] or r["deadline_fixed"]:
        bits.append(f"due: {r['deadline_text'] or r['deadline_fixed']}")
    if r["amount_kind"] != "none":
        bits.append(f"amount: {r['amount_kind']} {r['amount_value']} {r['amount_unit']} of {r['amount_base'] or '-'}")
    if r["penalty_text"]:
        bits.append(f"penalty: {r['penalty_text']}")
    return " | ".join(bits)


def stage_compile(job: Job, prog: Progress, force: bool = False) -> list[str]:
    if not force and C.get_process(process_key(job)) is not None:
        return []          # already compiled (and possibly reviewed): never silently overwrite
    keys = job.chunks["ckey"].tolist()
    tri, rel = C.markup_get(keys, "triage", job.model), C.relevance_get(keys, job.query_key, job.model)
    ex = C.markup_get(keys, "extract", job.model)
    regime = C.regime_get(regime_key(job))
    if not regime:
        return ["No regime outline; run the earlier stages."]
    by_code = {r["code"]: r for _, r in job.chunks.iterrows()}
    rules_rec, decisions, lines, decision_lines = [], [], [], []
    order = [c for p in sorted(regime["phases"], key=lambda p: PHASES.index(p["phase"])) for c in p["codes"]]
    phase_of = {c: p["phase"] for p in regime["phases"] for c in p["codes"]}
    cur = None
    for code in order:
        r = by_code[code]
        e = ex.get(r["ckey"], {})
        if phase_of[code] == "out_of_scope":
            continue
        if phase_of[code] != cur:
            cur = phase_of[code]
            lines.append(f"PHASE {cur}:")
        if e.get("kind") == "rates" and e.get("rows"):
            did = f"D{len(decisions) + 1}"
            decisions.append({"id": did, "name": e.get("table_title") or label(r["header"], 60), "chunk": code, "rows": e["rows"]})
            classes = ", ".join(x["class_label"] for x in e["rows"] if x["class_label"])[:160]
            decision_lines.append(f"{did} | rate schedule from {code} §{sec_of(r) or '?'} | {len(e['rows'])} rows | classes: {classes}")
        for k, rule in enumerate(e.get("rules", []), 1):
            rc = f"{code}.{k}"
            rules_rec.append({"code": rc, "chunk": code, **rule})
            lines.append(_rule_line(rc, r["level"], sec_of(r), rule))
    if not rules_rec and not decisions:
        return ["No extracted rules or tables to compile."]
    qi = C.query_get(job.query_key, job.model) or {"goal": job.query}
    user = (f"GOAL: {qi['goal']}\nJurisdiction: {job.place}, {job.state.upper()}\n\n" + "\n".join(lines)
            + ("\n\nDECISION TABLES available:\n" + "\n".join(decision_lines) if decision_lines else "")
            + ("\n\nMISSING from the sources (do not invent): " + "; ".join(m["item"] for m in regime.get("missing", [])) if regime.get("missing") else ""))
    prog.set(job.pid, stage="compile", done=0, total=1)
    graph = call_llm(job.model, job.host, COMPILE_SYSTEM, user, COMPILE_SCHEMA, stage="compile", ctx=job.query_key, timeout=job.timeout)
    chunk_recs = []
    for _, r in job.chunks.iterrows():
        chunk_recs.append({
            "code": r["code"], "key": r["ckey"], "header": clean(r["header"]), "section": sec_of(r), "level": r["level"],
            "source": r["source"], "citation": clean(r.get("citation")), "path": clean(r.get("path")),
            "source_url": clean(r.get("source_url")), "status": clean(r.get("status")), "fn": clean(r.get("fn")),
            "topic": clean(r.get("topic")), "text": clean(r["content"])[:2500],
            "role": tri.get(r["ckey"], {}).get("role", ""), "relevance": rel.get(r["ckey"], {}).get("verdict", ""),
            **{d: (None if pd.isna(r.get(d)) else float(r[d])) for d in DIM_LIST}})
    pkey = process_key(job)
    C.save_process(pkey, job.state, job.place, job.query, job.model, {
        "graph": graph, "chunks": chunk_recs, "rules": rules_rec, "decisions": decisions, "regime": regime,
        "query_interp": qi, "query_key": job.query_key, "model": job.model, "verifier": job.ver,
        "schema_ver": C.SCHEMA_VER, "state": job.state, "place": job.place, "query": job.query, "verdicts": {},
        "subject": regime.get("subject", "")})
    prog.set(job.pid, done=1)
    return []


def stage_verify(job: Job, prog: Progress, force: bool = False) -> list[str]:
    pkey = process_key(job)
    rec = C.get_process(pkey)
    if not rec:
        return ["No compiled process to verify."]
    rules_by = {r["code"]: r for r in rec["rules"]}
    chunks = {c["code"]: c for c in rec["chunks"]}
    nodes, _, _, _ = normalize_graph(rec["graph"], set(chunks), set(rules_by), {d["id"] for d in rec["decisions"]})
    cited = sorted({c for n in nodes.values() if n["type"] == "task" for c in n["rules"]})
    have = rec.get("verdicts", {})
    todo = [c for c in cited if force or c not in have]
    verdicts = dict(have)
    lock = threading.Lock()

    def one(code):
        rule, chunk = rules_by[code], chunks[rules_by[code]["chunk"]]
        d = call_llm(job.ver, job.host, VERIFY_SYSTEM,
                     f"TEXT:\n{chunk['text'][:3000]}\n\nCLAIM:\n{json.dumps(claim_of(rule), ensure_ascii=False)}",
                     VERIFY_SCHEMA, stage="verify", ctx=pkey, timeout=job.timeout)
        v, issues, quote = _e(d.get("verdict"), VERDICTS, "partial"), [_s(i, 160) for i in d.get("issues") or []][:4], _s(d.get("quote"), 300)
        q_ok = quote_in(quote, chunk["text"])
        if v == "supported" and not q_ok:
            v, issues = "partial", issues + ["supporting quote not found verbatim in the text"]
        if v == "supported" and rule.get("evidence") and not rule.get("evidence_ok"):
            v, issues = "partial", issues + ["extraction's own evidence quote not found in the text"]
        with lock:
            verdicts[code] = {"verdict": v, "issues": issues, "quote": quote, "quote_ok": q_ok, "by": job.ver}

    errs = _map(job, prog, "verify", todo, one)
    C.patch_process(pkey, {"verdicts": verdicts, "verifier": job.ver})
    return errs


STAGES = ["triage", "query", "relevance", "extract", "regime", "compile", "verify"]


def run_pipeline(job: Job, prog: Progress, stages: list[str] | None = None, force_verify: bool = False,
                 force_compile: bool = False) -> dict:
    """Run (or resume) the pipeline for one place × model. Cached stages are skipped, so this is cheap to re-run."""
    log = {"errors": [], "stages": []}
    fns = {"triage": stage_triage, "query": stage_query, "relevance": stage_relevance, "extract": stage_extract,
           "regime": stage_regime, "compile": lambda j, p: stage_compile(j, p, force_compile),
           "verify": lambda j, p: stage_verify(j, p, force_verify)}
    try:
        for s in stages or STAGES:
            prog.set(job.pid, stage=s)
            try:
                errs = fns[s](job, prog)
            except LLMUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                errs = [f"{s}: {e}"]
            log["stages"].append(s)
            log["errors"] += errs
            if s == "compile" and C.get_process(process_key(job)) is None:
                break
        prog.set(job.pid, stage="done")
    except LLMUnavailable as e:
        log["errors"].append(str(e))
        prog.set(job.pid, stage="stopped", error=str(e))
    log["pkey"] = process_key(job)
    return log


def run_jobs(jobs: list[Job], prog: Progress, parallel: bool = True) -> dict[str, dict]:
    """Run jobs; with parallel=True each distinct model gets its own thread so two models work at the same time
    (jobs for the same model run one after another, bounded by Job.workers per stage)."""
    groups: dict[str, list[Job]] = {}
    for j in jobs:
        groups.setdefault(j.model if parallel else "_all", []).append(j)
    results: dict[str, dict] = {}

    def seq(js):
        for j in js:
            results[j.pid] = run_pipeline(j, prog)

    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
        futs = [ex.submit(seq, js) for js in groups.values()]
        for f in futs:
            f.result()
    return results


def agreement_matrix(recs: dict[str, dict | None]) -> list[dict]:
    """Pairwise agreement for any number of models: one row per pair."""
    names = [m for m, r in recs.items() if r]
    out = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ag = agreement(recs[x], recs[y])
            if ag:
                out.append({"models": f"{x}  vs  {y}", "task labels shared": ag["task_label_overlap"],
                            "typed facts shared": ag["typed_fact_overlap"], "tasks": f"{ag['tasks_a']} / {ag['tasks_b']}",
                            "only in first": ", ".join(ag["facts_only_a"]), "only in second": ", ".join(ag["facts_only_b"])})
    return out


def agreement(a: dict | None, b: dict | None) -> dict:
    """How much do two processes (different models, same place) agree? Cheap, deterministic, label-level."""
    if not a or not b:
        return {}

    def labels(rec):
        nodes, _, _, _ = normalize_graph(rec["graph"], {c["code"] for c in rec["chunks"]}, {r["code"] for r in rec["rules"]})
        return {re.sub(r"\W+", " ", n["label"].lower()).strip() for n in nodes.values() if n["type"] == "task"}

    def facts(rec):
        out = set()
        for r in rec["rules"]:
            for k in ("amount_value", "deadline_days", "threshold_value", "penalty_max_value", "deadline_fixed"):
                if r.get(k):
                    out.add((k, r[k]))
        return out

    la, lb, fa, fb = labels(a), labels(b), facts(a), facts(b)
    jac = lambda x, y: round(len(x & y) / len(x | y), 2) if (x | y) else None
    return {"task_label_overlap": jac(la, lb), "typed_fact_overlap": jac(fa, fb), "tasks_a": len(la), "tasks_b": len(lb),
            "facts_only_a": sorted(f"{k}={v}" for k, v in fa - fb)[:8], "facts_only_b": sorted(f"{k}={v}" for k, v in fb - fa)[:8]}

