# Markup schema (v3)

The pipeline does **not** ask a model to turn raw sections into a diagram. Each section is marked up in several
small, constrained passes. Every pass is stored (SQLite, keyed by content hash + model + schema version), so the
markup is reusable data in its own right: for other topics, other places, the Wikibase, and evaluation.

Bump `SCHEMA_VER` in `locus_core.py` when a schema or prompt changes meaningfully; caches are keyed by it, so old
and new results never mix.

## Passes

| # | Pass | LLM? | Stored in | Answers |
|---|---|---|---|---|
| 0 | quality | no | inside triage | Is there any text here? (page headers, bare `§ 14-19`, "Reserved" are skipped — never sent to a model) |
| 1 | triage | yes | `markup` (pass=`triage`) | What *kind* of text is it, who is it for, which subject(s), which legal scheme, one-line summary, cross-references |
| 1b | query | yes | `query_interp` | What process is the researcher asking about (goal, audience, subjects) |
| 2 | relevance | yes | `relevance` | Is this section part of *that* process for *that* audience? (a reason is stored) |
| 3 | extract | yes | `markup` (pass=`extract`) | Typed facts, routed by triage role: **rules**, **rate table**, or **definitions** |
| 4 | regime | yes | `regimes` | Phase assignment, what was set aside, what a complete process normally has but the sources lack |
| 5 | compile | yes | `processes` | The graph (tasks, gateways, outcomes) from phase-ordered rules; rate tables become decision tables |
| 6 | verify | yes (ideally another model) | inside `processes.verdicts` | Does the source text state each cited rule? Must quote it; the quote is checked verbatim |

Everything an LLM saw and said is in `llm_log` (stage, model, prompt, raw response, seconds, error).

## Vocabularies

* **role**: `definition, applicability, duty, procedure, rate_schedule, exemption, penalty, admin_power, authority, heading_or_artifact, other`
* **audience**: `regulated_person, official, court, general`
* **phase**: `scope, preconditions, application, determination, issuance, term_and_renewal, ongoing_duties, violation_and_penalty, appeal, out_of_scope`
* **rule_type**: `duty, prohibition, permission, procedure_step, exemption, threshold, classification, penalty, power`
* **amount_kind**: `none, flat_fee, percent_of_base, per_unit, tiered`. A rate is **never** a flat fee ($0.36 per $100 of gross receipts is `percent_of_base`, value 0.36, base "gross receipts").
* **verdict**: `supported, partial, unsupported`
* **subject**: controlled slugs in the `subjects` table (seeded; new ones arrive as `proposed` and wait for approval in the Library tab). Each can carry a Wikibase QID. This is the join key that makes "dog licensing in A vs B" a query.

## Triage record

```json
{"role": "duty", "audience": "regulated_person", "subjects": ["business-license"], "regime": "business license tax",
 "summary": "You need a city business license before doing business.", "self_contained": true,
 "refs": [{"target": "14-19", "relation": "rate_in"}],
 "quality": {"words": 41, "real_words": 33, "artifact": false, "truncated": false, "reason": ""}}
```

## Extract record (role-routed)

`kind: "rules"` → up to 6 rules:

```json
{"rule_type": "duty", "actor": "Business owner", "action": "Obtain city business license", "condition": "", "applies_if": [],
 "deadline_text": "within 30 days", "deadline_days": "30", "deadline_fixed": "", "deadline_anchor": "",
 "amount_text": "", "amount_kind": "none", "amount_value": "", "amount_unit": "", "amount_base": "",
 "threshold_variable": "", "threshold_op": "", "threshold_value": "", "threshold_unit": "",
 "penalty_text": "", "penalty_kind": "none", "penalty_max_value": "", "penalty_unit": "", "renewal": "annual",
 "evidence": "shall obtain a city business license within 30 days", "evidence_ok": true}
```

`kind: "rates"` → `{table_title, rows:[{class_label, covers, amount_kind, amount_value, amount_unit, amount_base, min_receipts, max_receipts, note, evidence, evidence_ok}]}` (becomes a DMN decision table).

`kind: "defs"` → `{terms:[{term, meaning}]}`.

### Guards applied to every extraction (never trust model output)

* Numbers (`deadline_days`, `amount_value`, `threshold_value`, `penalty_max_value`, `min/max_receipts`) are kept only if that number literally appears in the section text (also `.36` and "36 cents" for 0.36).
* A `flat_fee` whose number cannot be grounded is downgraded to `none`: an unverifiable fee is not a fee.
* `evidence` must be a verbatim quote; `evidence_ok` records whether it is.
* Enumerations are validated; anything else falls back to a safe default.

## Regime outline

```json
{"title": "Business licensing", "subject": "business-license",
 "phases": [{"phase": "preconditions", "codes": ["c1"], "note": ""}, {"phase": "determination", "codes": ["c3"], "note": ""}],
 "excluded": [{"code": "c5", "reason": "only about what officials do internally"}],
 "missing": [{"item": "renewal due date", "why": "no section states it"}], "by": "llm"}
```

`missing` is the most useful field for coverage work: it names what a complete process normally specifies but the
sources did not.

## Process record

`{graph, chunks[], rules[], decisions[], regime, query_interp, model, verifier, verdicts{rule_code: {verdict, issues, quote, quote_ok, by}}, subject, ...}`.
Node `sources` cite rule codes (`c3.1`); nodes citing no rule are drawn grey; nodes whose rules the verifier rejects are red; end nodes never carry facts.
Compile never overwrites an existing process (so a reviewer's work is safe); use **Re-compile** deliberately.

## Cost

Per place and model, for N candidate sections with R judged relevant and V cited rules:
about `N` (triage) + `N` (relevance) + `R` (extract) + 3 (query, regime, compile) + `V` (verify) calls.
Fifteen sections is typically 40–60 calls. All cached; a re-run costs zero.

## Adding things

* **A role / phase / subject:** add it to the lists in `locus_markup.py` (roles/phases) or the Library tab (subjects), bump `SCHEMA_VER` if roles or phases changed.
* **A new rule field:** add it to `_RULE`, `norm_rules`, `claim_of`, and `node_facts`; bump `SCHEMA_VER`.
* **A different topic** (zoning, dog licensing, noise): nothing to change; the phases and roles are topic-neutral.
