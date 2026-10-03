"""locus_bpmn — deterministic BPMN 2.0 emitter (lanes = actors) for a stored process record."""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

import pandas as pd

from locus_core import DIM_LIST, SCHEMA_VER, clean, label, sid
from locus_markup import node_facts, normalize_graph

# (fill, stroke) for high / mid / low bins of a z-score
PALETTE = {
    "opacity": [("#fecaca", "#b91c1c"), ("#fef3c7", "#b45309"), ("#bbf7d0", "#15803d")],
    "_other": [("#bfdbfe", "#1d4ed8"), ("#eff6ff", "#3b82f6"), ("#ffffff", "#93c5fd")],
}
UNSOURCED = ("#f3f4f6", "#6b7280")
KIND_TAG = {"start": "bpmn:startEvent", "end": "bpmn:endEvent", "task": "bpmn:userTask", "gateway": "bpmn:exclusiveGateway"}
# verdict from the verifier overrides colour: (fill, stroke)
VERDICT_STYLE = {"unsupported": ("#fecaca", "#7f1d1d"), "partial": (None, "#d97706")}
FACT_KEYS = ("modality", "condition", "deadline_days", "deadline_fixed", "amount_kind", "amount_value", "amount_unit",
             "amount_base", "threshold", "penalty_text", "penalty_max_value", "penalty_unit", "renewal")
TW, TH, ROWH, COLW, BAND = 160, 84, 108, 230, 30


def bin_for(dim: str, z):
    if z is None or pd.isna(z):
        return None
    pal = PALETTE.get(dim, PALETTE["_other"])
    return pal[0] if z >= 0.5 else pal[2] if z <= -0.5 else pal[1]


def emit_bpmn(rec: dict, state: str, place: str, query: str, color_dim: str | None):
    """Return (xml, element_meta, warnings). Deterministic for a given stored record."""
    chunks = {c["code"]: c for c in rec["chunks"]}
    rules_by = {r["code"]: r for r in rec.get("rules", [])}
    decisions = {d["id"]: d for d in rec.get("decisions", [])}
    verdicts = rec.get("verdicts", {})
    nodes, order, edges, dropped = normalize_graph(rec["graph"], set(chunks), set(rules_by), set(decisions))
    warnings = [f"{dropped} edge(s) dropped (unknown or invalid node ids)."] if dropped else []
    if not nodes:
        return None, [], ["The model returned no nodes."]

    inc = {n: 0 for n in nodes}
    for e in edges:
        inc[e["tgt"]] += 1
    order = [n for n in order if not (nodes[n]["type"] == "end" and inc[n] == 0)]
    for n in [n for n in nodes if n not in order]:
        del nodes[n]
    S, E = "__start", "__end"
    nodes[S] = {"id": S, "type": "start", "actor": "", "label": "Start", "sources": [], "rules": [], "decision": ""}
    inc = {n: 0 for n in nodes}
    for e in edges:
        inc[e["tgt"]] += 1
    roots = [n for n in order if inc[n] == 0] or order[:1]
    edges = [{"src": S, "tgt": n, "label": ""} for n in roots] + edges
    outc = {n: 0 for n in nodes}
    for e in edges:
        outc[e["src"]] += 1
    need_end = [n for n in order if nodes[n]["type"] != "end" and (outc[n] == 0 or (nodes[n]["type"] == "gateway" and outc[n] < 2))]
    all_order = [S] + order
    if need_end:
        nodes[E] = {"id": E, "type": "end", "actor": "", "label": "End", "sources": [], "rules": [], "decision": ""}
        all_order.append(E)
        edges += [{"src": n, "tgt": E, "label": "Otherwise" if nodes[n]["type"] == "gateway" else ""} for n in need_end]

    adj = {n: [] for n in all_order}
    for e in edges:
        adj[e["src"]].append(e["tgt"])
    mark, back = {n: 0 for n in all_order}, set()

    def dfs(u):
        mark[u] = 1
        for v in adj[u]:
            if mark[v] == 0:
                dfs(v)
            elif mark[v] == 1:
                back.add((u, v))
        mark[u] = 2

    dfs(S)
    for n in all_order:
        if mark[n] == 0:
            edges.append({"src": S, "tgt": n, "label": ""})
            adj[S].append(n)
            dfs(n)
    if back:
        warnings.append(f"{len(back)} loop-back edge(s) drawn along the bottom.")
    fwd = [e for e in edges if (e["src"], e["tgt"]) not in back]
    indeg = {n: 0 for n in all_order}
    for e in fwd:
        indeg[e["tgt"]] += 1
    depth, queue, topo = {n: 0 for n in all_order}, [n for n in all_order if indeg[n] == 0], []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for e in fwd:
            if e["src"] == u:
                depth[e["tgt"]] = max(depth[e["tgt"]], depth[u] + 1)
                indeg[e["tgt"]] -= 1
                if indeg[e["tgt"]] == 0:
                    queue.append(e["tgt"])

    lanes: list[str] = []
    lane_of: dict[str, int] = {}
    canon: dict[str, str] = {}
    parent: dict[str, str] = {}
    for e in fwd:
        parent.setdefault(e["tgt"], e["src"])
    for n in topo:
        if n == S:
            continue
        a = nodes[n]["actor"]
        if a:
            a = canon.setdefault(a.lower(), a)
        elif parent.get(n) in lane_of:
            a = lanes[lane_of[parent[n]]]
        else:
            a = "Unassigned"
        if a not in lanes:
            lanes.append(a)
        lane_of[n] = lanes.index(a)
    first_child = next((e["tgt"] for e in fwd if e["src"] == S), None)
    lane_of[S] = lane_of.get(first_child, 0)
    if not lanes:
        lanes = ["Unassigned"]

    groups: dict[tuple[int, int], list[str]] = {}
    for n in topo:
        groups.setdefault((lane_of[n], depth[n]), []).append(n)
    lane_h = [max([len(v) for (l, _), v in groups.items() if l == li] or [1]) * ROWH + 20 for li in range(len(lanes))]
    lane_top = [sum(lane_h[:i]) for i in range(len(lanes))]
    maxd = max(depth.values())
    pool_w = BAND + 40 + (maxd + 1) * COLW + 40
    pool_h = sum(lane_h) + (40 if back else 0)
    size = {"start": (36, 36), "end": (36, 36), "gateway": (50, 50), "task": (TW, TH)}
    pos: dict[str, dict] = {}
    for (li, d), ns in groups.items():
        for row, n in enumerate(ns):
            w, h = size[nodes[n]["type"]]
            cx = BAND + 40 + d * COLW + TW / 2
            cy = lane_top[li] + 10 + row * ROWH + ROWH / 2
            pos[n] = {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h, "cy": cy}

    pid = sid("Process", state, place, query)
    nid = {n: sid("N", pid, n) for n in nodes}
    meta, node_xml = [], {}
    for n in topo:
        nd = nodes[n]
        srcs = [chunks[c] for c in nd["sources"]]
        facts = node_facts(nd, rules_by)
        vs = [verdicts.get(c, {}).get("verdict") for c in nd["rules"]]
        nd["_verdict"] = ("unsupported" if "unsupported" in vs else "partial" if "partial" in vs
                          else "supported" if vs and all(v == "supported" for v in vs) else "")
        if nd["type"] == "task" and nd["_verdict"] == "unsupported":
            warnings.append(f"Unsupported by its source: “{nd['label'][:60]}” (verifier said the text does not state it).")
        mean = {d: (float(pd.Series([s[d] for s in srcs if s.get(d) is not None]).mean()) if any(s.get(d) is not None for s in srcs) else None)
                for d in DIM_LIST}
        nd["_mean"] = mean
        if nd["type"] == "task" and not srcs:
            warnings.append(f"Unsourced task: “{nd['label'][:60]}” (drawn grey — treat as unverified).")
        shown = [("Modality", facts.get("modality")), ("Condition", facts.get("condition")),
                 ("Amount", " ".join(filter(None, [facts.get("amount_kind"), facts.get("amount_value"), facts.get("amount_unit"),
                                                   ("of " + facts["amount_base"]) if facts.get("amount_base") else ""]))),
                 ("Amount as written", facts.get("amount_text")), ("Threshold", facts.get("threshold")),
                 ("Deadline (days)", facts.get("deadline_days")), ("Due date", facts.get("deadline_fixed")),
                 ("Renewal", facts.get("renewal")), ("Penalty", facts.get("penalty_text")),
                 ("Max penalty", " ".join(filter(None, [facts.get("penalty_max_value"), facts.get("penalty_unit")]))),
                 ("Decision table", f"{decisions[nd['decision']]['id']} {decisions[nd['decision']]['name']}" if nd["decision"] else "")]
        details = "; ".join(f"{k}: {v}" for k, v in shown if v)
        ev = [f"[{c}] verdict={verdicts[c]['verdict']}" + (f" — “{verdicts[c]['quote']}”" if verdicts[c].get("quote") else "")
              for c in nd["rules"] if c in verdicts]
        doc = "\n\n".join(filter(None, [details, "\n".join(ev)] + [f"[§{s['section'] or '?'} {label(s['header'], 80)}] {s['text'][:700]}" for s in srcs]))
        attrs = {"actor": nd["actor"], "chunks": ",".join(s["key"] for s in srcs),
                 "sections": ", ".join(s["section"] for s in srcs if s["section"]),
                 **{k: facts.get(k) for k in FACT_KEYS},
                 "levels": ",".join(sorted({s["level"] for s in srcs})), "citations": " | ".join(s["citation"] for s in srcs if s.get("citation")),
                 "verdict": nd["_verdict"], "decision": nd["decision"],
                 **{d: (None if mean[d] is None else round(mean[d], 3)) for d in DIM_LIST},
                 "provenance": "llm-extracted" if srcs else "unsourced"}
        if n not in (S, E):
            meta.append({"id": nid[n], "type": nd["type"], "label": nd["label"], **{k: v for k, v in attrs.items() if v not in (None, "")}})
        node_xml[n] = (doc, attrs, srcs, bool(details or ev))

    o = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"',
        ' xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"',
        ' xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"',
        ' xmlns:di="http://www.omg.org/spec/DD/20100524/DI"',
        ' xmlns:bioc="http://bpmn.io/schema/bpmn/biocolor/1.0"',
        ' xmlns:lex="https://lexipedia.xyz/ns/bpmn/1.0"',
        f' id="{sid("Definitions", pid)}" targetNamespace="https://lexipedia.xyz/bpmn/locus">',
        f'  <bpmn:collaboration id="{pid}_collab">',
        f'    <bpmn:participant id="{pid}_part" name={quoteattr(clean(f"{place.title()}, {state.upper()} — {query}"))} processRef="{pid}"/>',
        "  </bpmn:collaboration>",
        f'  <bpmn:process id="{pid}" isExecutable="false">',
        f"    <bpmn:documentation>{escape(clean(rec['graph'].get('title', '')))}</bpmn:documentation>",
        "    <bpmn:extensionElements>",
        f'      <lex:jurisdiction state={quoteattr(state)} place={quoteattr(clean(place))} source="LocalLaws/LOCUS-v1" query={quoteattr(clean(query))}/>',
        f'      <lex:generator name="locus_explorer" version="0.4" model={quoteattr(clean(rec.get("model", "")))} schemaVersion={quoteattr(SCHEMA_VER)} verifier={quoteattr(clean(rec.get("verifier", "")))} reviewStatus={quoteattr(rec.get("status", "llm-draft"))}/>',
        "    </bpmn:extensionElements>",
        f'    <bpmn:laneSet id="{pid}_lanes">',
    ]
    for li, ln in enumerate(lanes):
        o.append(f'      <bpmn:lane id="{pid}_lane{li}" name={quoteattr(clean(ln))}>')
        o += [f"        <bpmn:flowNodeRef>{nid[n]}</bpmn:flowNodeRef>" for n in topo if lane_of[n] == li]
        o.append("      </bpmn:lane>")
    o.append("    </bpmn:laneSet>")
    fid = {i: sid("Flow", pid, e["src"], e["tgt"]) for i, e in enumerate(edges)}
    for n in topo:
        nd = nodes[n]
        doc, attrs, srcs, has_details = node_xml[n]
        tag = "bpmn:businessRuleTask" if nd.get("decision") else KIND_TAG[nd["type"]]
        o.append(f'    <{tag} id="{nid[n]}" name={quoteattr(clean(nd["label"]))}>')
        if srcs or has_details:
            o.append(f"      <bpmn:documentation>{escape(clean(doc))}</bpmn:documentation>")
        if n not in (S, E):
            a = " ".join(f"{k}={quoteattr(clean(v))}" for k, v in attrs.items() if v not in (None, ""))
            o.append(f"      <bpmn:extensionElements><lex:source {a}/></bpmn:extensionElements>")
        o += [f"      <bpmn:incoming>{fid[i]}</bpmn:incoming>" for i, e in enumerate(edges) if e["tgt"] == n]
        o += [f"      <bpmn:outgoing>{fid[i]}</bpmn:outgoing>" for i, e in enumerate(edges) if e["src"] == n]
        o.append(f"    </{tag}>")
    for i, e in enumerate(edges):
        nm = f" name={quoteattr(e['label'])}" if e["label"] else ""
        o.append(f'    <bpmn:sequenceFlow id="{fid[i]}" sourceRef="{nid[e["src"]]}" targetRef="{nid[e["tgt"]]}"{nm}/>')
    o.append("  </bpmn:process>")

    o += [f'  <bpmndi:BPMNDiagram id="{pid}_diagram">', f'    <bpmndi:BPMNPlane id="{pid}_plane" bpmnElement="{pid}_collab">',
          f'      <bpmndi:BPMNShape id="{pid}_part_di" bpmnElement="{pid}_part" isHorizontal="true">',
          f'        <dc:Bounds x="0" y="0" width="{pool_w}" height="{pool_h}"/></bpmndi:BPMNShape>']
    for li in range(len(lanes)):
        o += [f'      <bpmndi:BPMNShape id="{pid}_lane{li}_di" bpmnElement="{pid}_lane{li}" isHorizontal="true">',
              f'        <dc:Bounds x="{BAND}" y="{lane_top[li]}" width="{pool_w - BAND}" height="{lane_h[li]}"/></bpmndi:BPMNShape>']
    for n in topo:
        p, nd = pos[n], nodes[n]
        extra = ' isMarkerVisible="true"' if nd["type"] == "gateway" else ""
        col = ""
        if nd["type"] == "task":
            b = UNSOURCED if not nd["sources"] else (bin_for(color_dim, nd["_mean"].get(color_dim)) if color_dim else None)
            vs_ = VERDICT_STYLE.get(nd["_verdict"])
            if vs_ and nd["sources"]:
                b = (vs_[0] or (b[0] if b else "#ffffff"), vs_[1])
            if b:
                col = f' bioc:fill="{b[0]}" bioc:stroke="{b[1]}"'
        o += [f'      <bpmndi:BPMNShape id="{nid[n]}_di" bpmnElement="{nid[n]}"{extra}{col}>',
              f'        <dc:Bounds x="{p["x"]:.0f}" y="{p["y"]:.0f}" width="{p["w"]}" height="{p["h"]}"/></bpmndi:BPMNShape>']
    y_low = pool_h - 14
    for i, e in enumerate(edges):
        a, b = pos[e["src"]], pos[e["tgt"]]
        sx, sy, tx, ty = a["x"] + a["w"], a["cy"], b["x"], b["cy"]
        if (e["src"], e["tgt"]) in back:
            pts = [(sx, sy), (sx + 20, sy), (sx + 20, y_low), (tx - 20, y_low), (tx - 20, ty), (tx, ty)]
        elif abs(sy - ty) < 1:
            pts = [(sx, sy), (tx, ty)]
        else:
            mx = tx - 35
            pts = [(sx, sy), (mx, sy), (mx, ty), (tx, ty)]
        o.append(f'      <bpmndi:BPMNEdge id="{fid[i]}_di" bpmnElement="{fid[i]}">')
        o += [f'        <di:waypoint x="{x:.0f}" y="{y:.0f}"/>' for x, y in pts]
        o.append("      </bpmndi:BPMNEdge>")
    o += ["    </bpmndi:BPMNPlane>", "  </bpmndi:BPMNDiagram>", "</bpmn:definitions>"]
    return "\n".join(o), meta, warnings
