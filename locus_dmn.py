"""
locus_dmn — rate schedules -> DMN 1.3 decision tables (hit policy UNIQUE-ish, FIRST to stay safe).

Inputs  : business class (string), gross receipts (number)
Outputs : amount, basis
Rows come straight from the extracted rate table (no LLM here), so the table says exactly what was extracted;
verify it against the ordinance before relying on it.
"""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from locus_core import clean, sid


def _range(lo: str, hi: str) -> str:
    if lo and hi:
        return f"[{lo}..{hi}]"
    if lo:
        return f">={lo}"
    if hi:
        return f"<={hi}"
    return "-"


def decision_xml(dec: dict, place: str, state: str) -> str:
    did = sid("Decision", state, place, dec["id"], dec["name"])
    rows = []
    for i, r in enumerate(dec["rows"]):
        cls = f'"{clean(r["class_label"])}"' if r["class_label"] else "-"
        amt = r["amount_value"] or ""
        basis = " ".join(filter(None, [r["amount_kind"], r["amount_unit"], ("of " + r["amount_base"]) if r["amount_base"] else ""]))
        rows.append(f"""      <rule id="{did}_r{i}">
        <inputEntry id="{did}_r{i}_i0"><text>{escape(cls)}</text></inputEntry>
        <inputEntry id="{did}_r{i}_i1"><text>{escape(_range(r["min_receipts"], r["max_receipts"]))}</text></inputEntry>
        <outputEntry id="{did}_r{i}_o0"><text>{escape(amt or "null")}</text></outputEntry>
        <outputEntry id="{did}_r{i}_o1"><text>{escape(chr(34) + basis + chr(34))}</text></outputEntry>
        <description>{escape(clean(r["covers"] or r["note"]))}</description>
      </rule>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="{did}_defs" name={quoteattr(clean(dec['name']))}
             namespace="https://lexipedia.xyz/dmn/locus">
  <decision id="{did}" name={quoteattr(clean(dec['name']))}>
    <decisionTable id="{did}_table" hitPolicy="FIRST">
      <input id="{did}_in0" label="Business class"><inputExpression id="{did}_in0e" typeRef="string"><text>businessClass</text></inputExpression></input>
      <input id="{did}_in1" label="Gross receipts"><inputExpression id="{did}_in1e" typeRef="number"><text>grossReceipts</text></inputExpression></input>
      <output id="{did}_out0" label="Amount" name="amount" typeRef="number"/>
      <output id="{did}_out1" label="Basis" name="basis" typeRef="string"/>
{chr(10).join(rows)}
    </decisionTable>
  </decision>
</definitions>
"""
