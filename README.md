# LOCUS explorer

Explore [LocalLaws/LOCUS-v1](https://huggingface.co/datasets/LocalLaws/LOCUS-v1) (Peskoff et al., arXiv 2606.19334),
compare the same topic across municipalities, and turn local-ordinance text into reviewable BPMN process models
that can feed [Lexipedia](https://lexipedia.xyz).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b            # any model; configurable in the sidebar
streamlit run locus_explorer.py
```

Env vars: `HF_TOKEN` (if gated), `LOCUS_SRC` (parquet path / glob / `hf://` glob), `LOCUS_SLIM`, `LOCUS_DB`,
`OLLAMA_HOST`, `OLLAMA_MODEL`. The default source glob is a guess at the Hub layout; change it in the sidebar if needed.
Click **Build slim cache** once so the national views don't re-read 2.2M rows of text.

## What it does

| Tab | |
|---|---|
| National view | state choropleths of the four LOCUS dimensions, city-vs-county comparison |
| Jurisdiction | composition, dimensions by topic, hardest-to-read provisions |
| Provisions | searchable text browser |
| Process builder | annotate → compile → review for one place |
| Compare | same topic across 2–8 places (you choose how many): fee / deadline / renewal / penalty table + processes |

### The process pipeline

1. **Annotate** – Ollama reads each matching section (JSON-schema constrained, temperature 0, fixed seed) and
   extracts a plain-English summary and steps: actor, action, condition, deadline, fee, penalty, plus typed
   values (fee USD, deadline days, renewal, max penalty, deontic modality). Typed numbers are kept only if they
   literally appear in the source text.
2. **Compile** – a second pass assembles the steps into one graph (tasks, gateways, outcomes) ordered by deadlines,
   conditions and cross-references. Nodes cite step codes; unknown IDs and bad edges are dropped.
3. **Draw** – deterministic BPMN 2.0 with a lane per actor. Grey box = no source cited. Fill can show the mean
   LOCUS score (e.g. opacity) of the cited provisions.
4. **Review** – status `llm-draft → reviewed-ok / needs-work / rejected` stored in SQLite.
5. **Export (dry run)** – only `reviewed-ok` processes: item bundle + QuickStatements in two passes. See
   [docs/wikibase-mapping.md](docs/wikibase-mapping.md).

Everything the model produced is cached in SQLite keyed by chunk hash + model + prompt version, so a stored process
re-renders identically and nothing is re-run.

## Layout

```
locus_explorer.py    Streamlit UI + DuckDB access
locus_core.py        helpers, SQLite store, Ollama client, annotate/compile, matching
locus_bpmn.py        BPMN 2.0 emitter (lanes by actor)
lexipedia_export.py  bundle + QuickStatements; CLI: check | list | export
properties.json      Wikibase property map (existing + proposed)
tests/               pytest with a fake Ollama and a synthetic LOCUS-shaped parquet
```

## Tests

```bash
pytest -q
```

The tests run the whole flow against a fake Ollama server and synthetic data. They do **not** cover a real model's
output quality, the real dataset, the BPMN viewer in a browser, or a live Wikibase.

## Caveats

- A 7B model will misread some sections. Treat output as a draft for a lawyer; the plain-English summary and source
  text sit next to every step for that reason.
- Row order in LOCUS is assumed to be code order in one chart; verify.
- Long edges across several columns in the BPMN can pass behind a box.
- Check the dataset's licence before redistributing text derived from it.
