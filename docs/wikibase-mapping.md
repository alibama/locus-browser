# Mapping LOCUS-derived process models to lexipedia.wikibase.cloud

Status: **proposal**. Nothing is written to the instance. `properties.json` is the single source of truth for
property IDs and datatypes; edit it, then run `python lexipedia_export.py check` from a machine that can reach
the instance to compare it with what is actually there.

## What gets exported (and what doesn't)

Only processes whose review status is **reviewed-ok**. Only provisions that a process step actually cites.
Raw LOCUS chunks are not copied (2.2M rows; dataset licence not yet checked; Wikibase strings are short).
Provision items carry a citation, the section number, and the LOCUS chunk key, not the text.

| Item class | One per | Key statements |
|---|---|---|
| legal process model | compiled process | instance of, jurisdiction, regulated subject, based on (provisions), has part (norms), extraction record |
| legal norm (process step) | task node | instance of, jurisdiction, regulated subject, deontic modality, bearer, condition, sanction, fee amount, time limit, renewal interval, max penalty amount, based on, followed by (+ condition qualifier from gateways) |
| ordinance provision | cited chunk | instance of, jurisdiction, regulated subject, citation, legal identifier (section no.), LOCUS chunk ID |
| legal role | distinct actor | instance of |
| regulated subject | topic (e.g. "Dog licensing") | instance of — shared across places; this is the join key |

## Existing properties used

| Key | ID | Used for | Datatype (assumed unless checked) |
|---|---|---|---|
| instance_of | P5 | class membership | item |
| has_part | P10 | process → norms | item |
| followed_by | P12 | norm → next norm; qualifier P18 carries the gateway answer ("Licensed in time? → No") | item |
| based_on | P14 | process/norm → provisions | item |
| jurisdiction | P15 | everything | item |
| deontic_modality | P16 | obligation / prohibition / permission / power | item |
| bearer | P17 | the actor who must/may act | item |
| condition | P18 | trigger text; also qualifier on followed_by | string |
| sanction | P20 | penalty text as written | string |
| citation | P22 | provision citation | string |
| legal_identifier | P25 | section number | string |

Not used yet: P9 part of, P11 follows (inverse of P12; emit one direction only), P13 references (provision
cross-references are extracted but not exported yet), P19 governed by norm, P21 defeated by (natural home for
`exemption` sections), P23 official text and P24 effective date (LOCUS rows carry no URL or date).
P1–P4 (father/mother/child/image) look like seed properties and are untouched.

## Proposed new properties (7)

Kept to what the dog-licensing comparison needs. The first four are the essential ones.

| Key | Datatype | Why |
|---|---|---|
| **regulated subject** | item | Join key for "compare X across places". Without it nothing is queryable across jurisdictions. |
| **fee amount** | quantity (unit USD) | Sortable, comparable fees. |
| **time limit** | quantity (unit day) | Deadlines in days. |
| **renewal interval** | string | annual / biennial / one-time / none. Could become an item later. |
| maximum penalty amount | quantity (USD) | Typed ceiling; P20 keeps the text. |
| LOCUS chunk ID | external identifier | Traceability; lets re-runs find existing items. |
| extraction record | string | Model, prompt version, reviewer note, on every LLM-assisted process. |

## Items that must exist before pass 1

Class items (provision, norm, process, subject, legal role), unit items (USD, day), the four modality items, and
the jurisdiction item for each place (your Govdirectory pipeline already creates these). Put their QIDs in
`qid_map.json`; the exporter lists every key it could not resolve.

## Two passes

1. `pass1.qs`: CREATE every new item with label, description, an alias `lex-local:<key>` (so you can find the
   QID afterwards), and all statements whose value is a literal or an already-known item.
2. Record the QIDs pass 1 produced in `qid_map.json` (`{"prov:...": "Q123", ...}`), re-run, and `pass2.qs`
   contains the links between new items (has part, based on, followed by, bearer, regulated subject).

An item already present in `qid_map.json` is never re-created, so shared items (subject, legal roles) are created once.

## Questions the pilot should be able to answer from the Wikibase side

- Which of these places require annual renewal of a dog license, and what is the fee in each?
- What is the deadline (days) and the maximum fine?
- Which provision does each step come from, and who reviewed it?

If someone who is not the author can answer those from the items, the model is good enough to scale.

## Open questions

- Datatypes of P15–P25 were not verified (the instance sits behind a bot challenge when fetched automatically).
- Intended meaning of P19 and P21 in your model; this proposal does not use them.
- Whether `renewal interval` and `condition` should be items rather than strings.
- Licence terms of LocalLaws/LOCUS-v1 for redistributing text derived from it.
