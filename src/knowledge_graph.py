"""
Clinical knowledge graph for the Sep Agent (opt-in, additive).

Loads a human-editable JSON source (clinical_kg.json) into a networkx directed
graph, and offers structured reasoning the text-only RAG layer cannot:
  - `diagnoses_for_findings`  : which diagnoses a set of agent findings reach,
    with the reasoning PATH (finding -> criterion -> diagnosis) for traceability.
  - `convergent_diagnoses`    : diagnoses supported by findings from >= 2 DIFFERENT
    modality agents - explicit CROSS-MODALITY convergence (the shipped consumer),
    which concatenating findings in the prompt cannot surface.

This graph is also the SINGLE SOURCE of the Sep Agent's directional facts: finding
nodes carry a `directional` block that sep_agent._load_directional_flags_config reads
(with plain json, no networkx) to drive its consistency check + prompt anchor - the
former separate directional_flags_config.json was migrated into this file.

Design: networkx (light, local, no server) over a JSON source (auditable,
version-controllable) - same spirit as the RAG index. Neo4j/RDF would be the
production choice at scale but are unnecessary here. The graph-traversal functions
are imported by the Sep Agent only when use_kg=True. Requires `pip install networkx`.
"""
import json
import os
from typing import Dict, List, Optional

import networkx as nx

_KG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_kg.json")

# Edge relations that carry evidence forward from a finding toward a diagnosis.
_FORWARD_RELATIONS = {"meets", "indicates", "supports"}


def load_kg(path: str = _KG_PATH) -> nx.DiGraph:
    """Load clinical_kg.json into a networkx DiGraph (node attrs + edge 'relation')."""
    data = json.load(open(path, encoding="utf-8"))
    g = nx.DiGraph()
    for n in data["nodes"]:
        node_id = n["id"]
        g.add_node(node_id, **{k: v for k, v in n.items() if k != "id"})
    for e in data["edges"]:
        g.add_edge(e["source"], e["target"], relation=e.get("relation", ""))
    return g


def _nodes_of_type(g: nx.DiGraph, node_type: str) -> set:
    return {n for n, d in g.nodes(data=True) if d.get("type") == node_type}


def reasoning_path(g: nx.DiGraph, finding_id: str, diagnosis_id: str) -> Optional[List[str]]:
    """Shortest forward path finding -> ... -> diagnosis (list of node labels), or None."""
    if finding_id not in g or diagnosis_id not in g:
        return None
    try:
        path = nx.shortest_path(g, finding_id, diagnosis_id)
    except nx.NetworkXNoPath:
        return None
    return [g.nodes[n].get("label", n) for n in path]


def _reached_nodes_by_type(g: nx.DiGraph, finding_ids: List[str], node_type: str) -> Dict[str, dict]:
    """
    Generic: for a set of active finding ids, return the nodes of `node_type` they
    reach (following the directed edges): {node_id: {"label", "supporting_findings":
    [...], "agents": set, "paths": {finding_id: [label,...]}}}.
    """
    target_nodes = _nodes_of_type(g, node_type)
    present = [f for f in finding_ids if f in g]
    out: Dict[str, dict] = {}
    for f in present:
        for d in nx.descendants(g, f) & target_nodes:
            entry = out.setdefault(d, {"label": g.nodes[d].get("label", d),
                                       "supporting_findings": [], "agents": set(),
                                       "paths": {}})
            entry["supporting_findings"].append(f)
            agent = g.nodes[f].get("agent")
            if agent:
                entry["agents"].add(agent)
            entry["paths"][f] = reasoning_path(g, f, d)
    return out


def diagnoses_for_findings(g: nx.DiGraph, finding_ids: List[str]) -> Dict[str, dict]:
    """Diagnoses (specific) reached by a set of active finding ids. See
    _reached_nodes_by_type."""
    return _reached_nodes_by_type(g, finding_ids, "diagnosis")


def convergent_diagnoses(g: nx.DiGraph, finding_ids: List[str]) -> Dict[str, dict]:
    """
    Cross-modality convergence: only the diagnoses supported by findings from >= 2
    DIFFERENT modality agents. Same shape as diagnoses_for_findings.
    """
    return {d: e for d, e in diagnoses_for_findings(g, finding_ids).items()
            if len(e["agents"]) >= 2}


# --- Hierarchy (pathology taxonomy) ---
# `is_a` edges roll a specific diagnosis up to its parent CATEGORY (the PTB-XL
# diagnostic superclasses CD/HYP/MI/STTC/NORM, plus clinical categories like heart
# failure). This lets the KG reason at the right level of abstraction and bridges
# the deterministic findings to the learned-embedding superclass flags, which point
# straight at the category nodes.

def category_of(g: nx.DiGraph, node_id: str) -> List[str]:
    """The parent category label(s) a diagnosis/finding rolls up to via `is_a`
    (and finding->diagnosis->is_a) edges. Empty if none."""
    if node_id not in g:
        return []
    cats = nx.descendants(g, node_id) & _nodes_of_type(g, "category")
    return sorted(g.nodes[c].get("label", c) for c in cats)


def categories_for_findings(g: nx.DiGraph, finding_ids: List[str]) -> Dict[str, dict]:
    """Parent categories reached by a set of findings (rolls specific findings up to
    the pathology superclass they belong to). Same shape as diagnoses_for_findings."""
    return _reached_nodes_by_type(g, finding_ids, "category")


def convergent_categories(g: nx.DiGraph, finding_ids: List[str]) -> Dict[str, dict]:
    """Categories reached by findings from >= 2 DIFFERENT modality agents -
    cross-modality convergence at the pathology-superclass level (more abstract than
    convergent_diagnoses; can catch agreement between a specific diagnosis in one
    modality and a superclass pattern in another)."""
    return {c: e for c, e in categories_for_findings(g, finding_ids).items()
            if len(e["agents"]) >= 2}


# NOTE - the Sep Agent's directional-consistency check does NOT live here. It stays
# in sep_agent (_check_directional_consistency, with its battle-tested ventricle-
# referent resolution), but now SOURCES its per-flag facts from this graph's finding
# nodes' `directional` blocks (via sep_agent._load_directional_flags_config reading
# clinical_kg.json). So the KG is the single source of the direction facts, while the
# proven check logic is unchanged - see that module. An earlier standalone KG check
# was removed as detection-equivalent and superseded by this migration.
