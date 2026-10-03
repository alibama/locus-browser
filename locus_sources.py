"""
locus_sources — adapters that put LOCUS-v1 (local ordinances) and vaquill/open-us-law (state + federal
statutes, regulations, ...) into ONE provision shape, so the rest of the pipeline never cares where a
section came from.

Unified columns (superset of what the pipeline uses):
    source        'locus' | 'open-us-law'
    level         'local' | 'state' | 'federal'
    state, place  place = city/county for local, 'statewide' for state, 'federal' for federal
    city, county  (LOCUS only)
    corpus        ordinances | statutes | regulations | court_rules | ...
    document_type
    citation, section_no, header (citation + title), path (breadcrumb), content (text)
    status        in_force | repealed | ...        source_url, year
    fn, topic, jtype, opacity, enforcement_discretion, paternalism, problem_salience   (LOCUS only; NULL for the rest)

open-us-law is gated on the Hub: accept its conditions and set HF_TOKEN. Its files are named
us_{jurisdiction}_{corpus}.parquet, so a state's law is a cheap glob rather than a 4 GB scan. Jurisdiction
and corpus are derived from the file name, so we don't depend on how the `state` column is spelled.
"""
from __future__ import annotations

import re

import pandas as pd

from locus_core import DIM_LIST

OUL_TEMPLATE = "hf://datasets/vaquill/open-us-law/us_{jur}_{corpus}.parquet"
OUL_CORPORA = ["statutes", "regulations", "court_rules", "agency_guidance", "constitutions"]
FEDERAL_CODES = ("us", "federal", "usc", "fed")
UNIFIED = ["source", "level", "state", "place", "city", "county", "corpus", "document_type", "citation", "section_no",
           "header", "path", "content", "status", "source_url", "year", "is_substantive", "fn", "topic", "jtype", *DIM_LIST]


def lit(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def ensure_unified(df: pd.DataFrame) -> pd.DataFrame:
    for c in UNIFIED:
        if c not in df.columns:
            df[c] = None
    return df[UNIFIED + [c for c in df.columns if c not in UNIFIED]]


def columns_of(cur, path: str) -> set[str]:
    rows = cur.execute(f"DESCRIBE SELECT * FROM read_parquet({lit(path)}, union_by_name=true)").fetchall()
    return {r[0] for r in rows}


# ───────────────────────────── LOCUS ─────────────────────────────
def locus_states(cur, path: str) -> list[str]:
    return [r[0] for r in cur.execute(
        f"SELECT DISTINCT state FROM read_parquet({lit(path)}, union_by_name=true) WHERE state IS NOT NULL ORDER BY 1").fetchall()]


def locus_places(cur, path: str, state: str) -> pd.DataFrame:
    return cur.execute(
        f"""SELECT DISTINCT coalesce(city, county) AS place, source_jurisdiction_type AS jtype
            FROM read_parquet({lit(path)}, union_by_name=true) WHERE state = ? AND coalesce(city, county) IS NOT NULL ORDER BY 1""",
        [state]).df()


def load_locus_place(cur, path: str, state: str, place: str) -> pd.DataFrame:
    dims = ", ".join(DIM_LIST)
    df = cur.execute(
        f"""SELECT 'locus' AS source, 'local' AS level, state, coalesce(city, county) AS place, city, county,
                   'ordinances' AS corpus, 'ordinance' AS document_type, NULL AS citation, NULL AS section_no,
                   header, NULL AS path, content, 'in_force' AS status, NULL AS source_url, NULL AS year,
                   is_substantive, "function" AS fn, topic, source_jurisdiction_type AS jtype, {dims}
            FROM read_parquet({lit(path)}, union_by_name=true) WHERE state = ? AND coalesce(city, county) = ?""",
        [state, place]).df().reset_index(drop=True)
    return ensure_unified(df)


# ───────────────────────────── open-us-law ─────────────────────────────
def oul_paths(template: str, jur: str, corpora: list[str]) -> list[str]:
    return [template.format(jur=jur.lower(), corpus=c) for c in corpora]


def _pick(cols: set[str], *names: str) -> str:
    for n in names:
        if n in cols:
            return n
    return "NULL"


def search_oul(cur, path: str, jur: str, terms: list[str], cap: int, pinned: list[str] | None = None,
               statuses: tuple[str, ...] = ("in_force",)) -> pd.DataFrame:
    """Top-`cap` sections of one open-us-law file matching ALL terms (title hits weigh more), plus any whose
    citation contains a pinned substring. Ranking is deliberately dumb; the LLM relevance pass does the real filtering."""
    cols = columns_of(cur, path)
    c = lambda *n: _pick(cols, *n)
    title, text, cit, status = c("section_title", "title_name"), c("text"), c("citation", "citation_short"), c("act_status")
    corpus = re.sub(r"^.*/us_[a-z]+_", "", path).replace(".parquet", "")
    level = "federal" if jur.lower() in FEDERAL_CODES else "state"
    t_expr = f"lower(coalesce(CAST({title} AS VARCHAR), ''))"
    x_expr = f"lower(coalesce(CAST({text} AS VARCHAR), ''))"
    params: list = []
    where = []
    for t in terms:
        where.append(f"({t_expr} LIKE ? OR {x_expr} LIKE ?)")
        params += [f"%{t.lower()}%", f"%{t.lower()}%"]
    cond = " AND ".join(where) if where else "FALSE"
    score = " + ".join(f"(CASE WHEN {t_expr} LIKE ? THEN 3 ELSE 0 END)" for t in terms) or "0"
    score_params = [f"%{t.lower()}%" for t in terms]
    pin_cond, pin_params = "FALSE", []
    if pinned:
        pin_cond = " OR ".join(f"lower(coalesce(CAST({cit} AS VARCHAR), '')) LIKE ?" for _ in pinned)
        pin_params = [f"%{p.lower()}%" for p in pinned]
    st_cond = ""
    if status != "NULL" and statuses:
        st_cond = f"AND (CAST({status} AS VARCHAR) IN ({', '.join(lit(s) for s in statuses)}) OR {status} IS NULL)"
    sql = f"""
        SELECT 'open-us-law' AS source, '{level}' AS level, {lit(jur.lower())} AS state,
               {lit('federal' if level == 'federal' else 'statewide')} AS place, NULL AS city, NULL AS county,
               {lit(corpus)} AS corpus, {c('document_type')} AS document_type,
               {cit} AS citation, {c('section_number')} AS section_no,
               trim(coalesce(CAST({cit} AS VARCHAR), '') || ' ' || coalesce(CAST({title} AS VARCHAR), '')) AS header,
               {c('display_path', 'breadcrumb')} AS path, {text} AS content, {status} AS status,
               {c('source_url')} AS source_url, {c('year', 'last_amended_year')} AS year,
               TRUE AS is_substantive, NULL AS fn, NULL AS topic, NULL AS jtype,
               {', '.join('CAST(NULL AS DOUBLE) AS ' + d for d in DIM_LIST)},
               ({pin_cond}) AS _pinned, ({score}) AS _score
        FROM read_parquet({lit(path)}, union_by_name=true)
        WHERE (({cond}) OR ({pin_cond})) {st_cond}
        ORDER BY _pinned DESC, _score DESC
        LIMIT {int(cap)}"""
    df = cur.execute(sql, [*pin_params, *score_params, *params, *pin_params]).df()
    return ensure_unified(df.drop(columns=["_pinned", "_score"], errors="ignore").reset_index(drop=True))


def load_law_context(cur, template: str, jur: str, corpora: list[str], terms: list[str], cap: int,
                     pinned: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Search each requested corpus file for one jurisdiction; collect per-file problems instead of failing."""
    frames, problems = [], []
    for path in oul_paths(template, jur, corpora):
        try:
            frames.append(search_oul(cur, path, jur, terms, cap, pinned))
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:180]
            problems.append(f"{path.rsplit('/', 1)[-1]}: {msg}")
    if not frames:
        return ensure_unified(pd.DataFrame(columns=UNIFIED)), problems
    df = pd.concat(frames, ignore_index=True).head(cap)
    return df, problems
