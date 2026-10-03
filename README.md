# LOCUS explorer

Explore local ordinances ([LocalLaws/LOCUS-v1](https://huggingface.co/datasets/LocalLaws/LOCUS-v1)) together with state and
federal law ([vaquill/open-us-law](https://huggingface.co/datasets/vaquill/open-us-law)), compare the same topic across
places and across **two models at once**, and turn the text into marked-up, verified, reviewable BPMN/DMN process models that
can feed [Lexipedia](https://lexipedia.xyz).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...            # open-us-law is gated: accept its conditions on the Hub first
ollama pull qwen2.5:7b         # or pull from the app's Models panel
streamlit run locus_explorer.py
```

Env: `HF_TOKEN`, `LOCUS_SRC`, `LOCUS_SLIM`, `LOCUS_DB`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OUL_TEMPLATE`.
The default LOCUS glob and open-us-law file template (`us_{jurisdiction}_{corpus}.parquet`) come from the dataset cards, not from
a directory listing; both are editable in the sidebar.

## What's in the app

| Tab | |
|---|---|
| National view | state choropleths of the four LOCUS dimensions, city-vs-county (LOCUS only) |
| Jurisdiction / Provisions | composition, dimensions, hardest-to-read sections, text browser |
| Process builder | one place, one or two models: markup → relevance → extract → outline → compile → verify |
| Compare | N places (2-8) × 1-2 models, side by side, with a fee/rate/deadline/penalty table and a model-agreement table |
| Library | controlled subject registry (links to Wikibase QIDs), stored processes, debug bundles |

### Models panel (sidebar)

* Shows whether Ollama is reachable; **Start Ollama here** (local only) and an install link if it is not.
* Lists installed models; for any model you type that is missing, **Pull** streams `ollama pull` with a progress bar.
* **Self-test** checks that a model can follow a JSON schema at all (small models sometimes cannot).
* **Model B** runs at the same time as Model A (one worker thread per model); an optional **verifier** model checks the extractor's work.
* Everything stored stays viewable with the server down; only running new stages needs a model.

### The markup pipeline

See [docs/markup-schema.md](docs/markup-schema.md). In short: deterministic quality check → triage (role, audience, subjects, regime,
summary) → relevance for *your* topic (with a stored reason) → role-routed typed extraction (rules / rate tables / definitions, each
with a verbatim evidence quote) → regime outline (phases, excluded, **missing**) → compile → verify (the verifier must quote the text;
the quote is checked). Rates are never flat fees; numbers must appear in the source; unsupported steps turn red.

### Sending samples back

Every process has a **Debug bundle** download (all stage outputs plus every prompt and raw model response). That is the thing to
send when a result looks wrong.

## Layout

```
locus_explorer.py    Streamlit UI
locus_core.py        helpers + SQLite store (markup, relevance, regimes, processes, subjects, llm_log)
locus_sources.py     LOCUS + open-us-law -> one provision shape
locus_llm.py         Ollama client, status/start/pull/self-test, call logging
locus_markup.py      vocabularies, schemas, prompts, stages, parallel runner, model agreement
locus_bpmn.py        BPMN 2.0 emitter (lanes by actor, verdict colouring)
locus_dmn.py         rate tables -> DMN decision tables
lexipedia_export.py  bundle + QuickStatements; CLI: check | list | export
properties.json      Wikibase property map (existing + proposed)
evals/               lint_process.py + a gold template (see below)
tests/               pytest with a fake Ollama and synthetic data
docs/                markup-schema.md, wikibase-mapping.md
```

## Tests and evals

```bash
pytest -q
python evals/lint_process.py --bpmn x.bpmn --sidecar x.json [--bundle b.json] [--pass1 p1.qs] [--gold evals/gold/....json]
```

`pytest` runs the whole pipeline against a fake Ollama: two models in flight at once, the pull flow, an unreachable server, grounding,
verification downgrades, DMN/BPMN well-formedness, the export gate. It does **not** measure a real model's quality, run against the real
datasets, or exercise the BPMN viewer in a browser. The lint finds known failure patterns; the gold file is an unverified reference
that needs a lawyer's sign-off.

## Caveats

* A small model will misread some sections. Treat output as a draft; summaries, evidence quotes and verdicts sit next to every step.
* open-us-law file names for federal law are not known to me; set the code in the sidebar after checking the Files tab.
* Retrieval of state law is keyword-and-title ranked, then the LLM relevance pass filters; use "Always include citations containing"
  to pin a known authority.
* LOCUS's licence still needs checking before redistributing derived text. open-us-law is CC BY 4.0 for the compilation, with public-domain text.
* Long edges across columns in the BPMN can pass behind a box.
