#!/usr/bin/env python3
import json
import re
import string
from pathlib import Path
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Iterable

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "capnav_qa_results"
DEFAULT_JSONL = RESULTS_DIR / "test.jsonl"

GT_ROOT = Path("/mmfs1/gscratch/makelab/ruiqi/datasets/capnav/ground_truth")

OUT_PER_RECORD = RESULTS_DIR / "scored_per_record.jsonl"
OUT_SUMMARY = RESULTS_DIR / "scored_summary.json"

_SCENE_CACHE: Dict[str, Tuple[Dict, Dict]] = {}

# =========================================================
# IO: jsonl
# =========================================================
def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Non-dict JSONL entry in {path} line {ln}: {type(obj)}")
            yield obj

def choose_input_files() -> List[Path]:
    """
    Default: read DEFAULT_JSONL
    Fallback: if missing, read all *.jsonl under capnav_qa_results
    """
    if DEFAULT_JSONL.exists():
        return [DEFAULT_JSONL]
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    return files

# =========================================================
# Text normalization
# =========================================================
def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

# =========================================================
# Parse VLM output: Answer + Path (+ optional Fail/Reason)
# =========================================================
def parse_vlm_output_text(output: str) -> Dict:
    if not isinstance(output, str) or not output.strip():
        return {"ok": False, "error": "empty_output", "raw": output}

    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    # Accept: Answer: (A|B) True|False | Path: ...
    # Optional tail like "| Fail: ..." is allowed (ignored here, but retained as raw)
    m = re.search(
        r"Answer:\s*\((A|B)\)\s*(True|False)\s*\|\s*Path:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if not m:
        return {"ok": False, "error": "cannot_parse_answer_path", "raw": cleaned}

    answer = "yes" if m.group(2).lower() == "true" else "no"
    path_str = m.group(3).strip()

    # If the model included extra fields after path, cut them off conservatively.
    # e.g., "a -> b | Fail: xxx"  -> keep "a -> b"
    path_str = re.split(r"\s*\|\s*(?:Fail|Reason)\s*:", path_str, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    path_names = [p.strip() for p in re.split(r"\s*(?:->|→)\s*", path_str) if p.strip()]

    if answer == "yes" and len(path_names) < 2:
        return {"ok": False, "error": "yes_without_path", "raw": cleaned}

    return {"ok": True, "answer": answer, "path_names": path_names, "raw": cleaned}

# =========================================================
# Reverse match src/tgt from question via graph node names
# =========================================================
def find_from_to_indices(qn: str) -> Optional[Tuple[int, int]]:
    m_from = re.search(r"\bfrom\b", qn)
    if not m_from:
        return None
    m_to = re.search(r"\bto\b", qn[m_from.end():])
    if not m_to:
        return None
    to_start = m_from.end() + m_to.start()
    return (m_from.end(), to_start)

def all_node_hits_in_question(qn: str, node_names: List[str]) -> List[Dict]:
    hits = []
    for name in node_names:
        nn = norm_text(name)
        if not nn:
            continue
        for m in re.finditer(rf"\b{re.escape(nn)}\b", qn):
            hits.append({
                "name": name,
                "start": m.start(),
                "end": m.end(),
                "length": len(nn),
                "center": (m.start() + m.end()) / 2.0
            })
    return hits

def derive_src_tgt_by_reverse_match(question: str, name_to_id: Dict[str, str]) -> Dict:
    if not isinstance(question, str) or not question.strip():
        return {"ok": False, "error": "empty_question"}

    qn = norm_text(question)
    node_names = list(name_to_id.keys())

    hits = all_node_hits_in_question(qn, node_names)
    if not hits:
        return {"ok": False, "error": "no_node_name_hit_in_question"}

    span_idx = find_from_to_indices(qn)
    if not span_idx:
        hits_sorted = sorted(hits, key=lambda h: (h["start"], -h["length"]))
        start_hit = hits_sorted[0]
        end_hit = hits_sorted[-1]
        if start_hit["name"] == end_hit["name"] and len(hits_sorted) >= 2:
            end_hit = hits_sorted[-2]
        return {
            "ok": True,
            "src_name": start_hit["name"],
            "tgt_name": end_hit["name"],
            "src_id": name_to_id[start_hit["name"]],
            "tgt_id": name_to_id[end_hit["name"]],
            "method": "fallback_first_last",
        }

    from_end, to_start = span_idx
    start_cands = [h for h in hits if from_end <= h["center"] <= to_start]
    end_cands = [h for h in hits if h["center"] >= to_start]

    def score_start(h):
        return (-abs(h["center"] - from_end) + 0.08 * h["length"])

    def score_end(h):
        return (-abs(h["center"] - to_start) + 0.08 * h["length"])

    if start_cands and end_cands:
        start_hit = max(start_cands, key=score_start)
        end_hit = max(end_cands, key=score_end)
        if start_hit["name"] == end_hit["name"] and len(end_cands) > 1:
            end_sorted = sorted(end_cands, key=score_end, reverse=True)
            for h in end_sorted:
                if h["name"] != start_hit["name"]:
                    end_hit = h
                    break
        return {
            "ok": True,
            "src_name": start_hit["name"],
            "tgt_name": end_hit["name"],
            "src_id": name_to_id[start_hit["name"]],
            "tgt_id": name_to_id[end_hit["name"]],
            "method": "from_to_partition",
        }

    after_from = [h for h in hits if h["center"] >= from_end]
    if not after_from:
        return {"ok": False, "error": "no_hits_after_from"}

    start_hit = min(after_from, key=lambda h: abs(h["center"] - from_end) - 0.05 * h["length"])
    end_hit = min(after_from, key=lambda h: abs(h["center"] - to_start) - 0.05 * h["length"])
    if start_hit["name"] == end_hit["name"]:
        alt = sorted(after_from, key=lambda h: abs(h["center"] - to_start) - 0.05 * h["length"])
        for h in alt:
            if h["name"] != start_hit["name"]:
                end_hit = h
                break

    return {
        "ok": True,
        "src_name": start_hit["name"],
        "tgt_name": end_hit["name"],
        "src_id": name_to_id[start_hit["name"]],
        "tgt_id": name_to_id[end_hit["name"]],
        "method": "degraded_closest",
    }

# =========================================================
# Load graph + traverse
# =========================================================
def load_graph_traverse(scene: str) -> Tuple[Dict, Dict]:
    if scene in _SCENE_CACHE:
        return _SCENE_CACHE[scene]

    graph_path = GT_ROOT / "graphs" / f"{scene}-graph.json"
    traverse_path = GT_ROOT / "traverse" / f"{scene}-traverse.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    if not traverse_path.exists():
        raise FileNotFoundError(f"Missing traverse: {traverse_path}")

    graph = json.loads(graph_path.read_text())
    traverse = json.loads(traverse_path.read_text())
    _SCENE_CACHE[scene] = (graph, traverse)
    return graph, traverse

def build_graph_index(graph: Dict) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    name_to_id = {n["name"]: n["id"] for n in graph["nodes"]}
    adj = {n["id"]: [] for n in graph["nodes"]}
    edge_set = set()
    for e in graph["edges"]:
        u, v = e["from"], e["to"]
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))  # undirected
    return name_to_id, adj, edge_set

# =========================================================
# Traversability (RTA)
# =========================================================
def agent_key(traverse: Dict, agent: str) -> Optional[str]:
    a = agent.upper()
    if a in traverse:
        return a
    if a == "HUMANOID" and "ROBOT" in traverse:
        return "ROBOT"
    return None

def edge_traversable(traverse: Dict, agent: str, u: str, v: str) -> bool:
    k = agent_key(traverse, agent)
    if not k:
        return False
    ad = traverse[k]
    # undirected lookup (OK per your note)
    return bool(ad.get(f"{u}|{v}", ad.get(f"{v}|{u}", {})).get("traversable", False))

def traversability_score(traverse: Dict, agent: str, path_ids: List[str]) -> Optional[float]:
    if not path_ids or len(path_ids) < 2:
        return None
    total = len(path_ids) - 1
    ok = 0
    for i in range(total):
        if edge_traversable(traverse, agent, path_ids[i], path_ids[i + 1]):
            ok += 1
    return ok / total

# =========================================================
# PV validity
# =========================================================
def pv_strict(path_ids: List[Optional[str]], src_id: str, tgt_id: str, edge_set: set) -> Tuple[bool, List[str]]:
    errors = []
    if not path_ids:
        return False, ["empty_path"]
    if any(x is None for x in path_ids):
        errors.append("unresolved_node_in_path")
    if path_ids[0] != src_id:
        errors.append("path_start_not_src")
    if path_ids[-1] != tgt_id:
        errors.append("path_end_not_tgt")

    clean = [x for x in path_ids if x is not None]
    if len(clean) != len(set(clean)):
        errors.append("not_simple_path_repeated_nodes")

    for i in range(len(path_ids) - 1):
        u, v = path_ids[i], path_ids[i + 1]
        if u is None or v is None:
            continue
        if (u, v) not in edge_set:
            errors.append(f"missing_edge:{u}->{v}")

    return (len(errors) == 0), errors

# =========================================================
# GT doable: exists traversable path (agent-conditioned BFS)
# =========================================================
def exists_traversable_path(adj: Dict[str, List[str]], traverse: Dict, agent: str, src: str, tgt: str) -> bool:
    if src == tgt:
        return True
    dq = deque([src])
    seen = {src}
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v in seen:
                continue
            if not edge_traversable(traverse, agent, u, v):
                continue
            if v == tgt:
                return True
            seen.add(v)
            dq.append(v)
    return False

# =========================================================
# Map predicted path node names -> node ids (conservative)
# =========================================================
def best_match_pred_name(pred_name: str, node_names: List[str]) -> Optional[str]:
    if not pred_name:
        return None
    pn = norm_text(pred_name)
    if not pn:
        return None

    for n in node_names:
        if norm_text(n) == pn:
            return n

    contain = []
    for n in node_names:
        nn = norm_text(n)
        if nn and (nn in pn or pn in nn):
            contain.append((len(nn), n))
    if contain:
        contain.sort(reverse=True)
        return contain[0][1]

    best = (0.0, None)
    for n in node_names:
        nn = norm_text(n)
        r = SequenceMatcher(None, pn, nn).ratio()
        if r > best[0]:
            best = (r, n)
    if best[0] >= 0.78:
        return best[1]
    return None

# =========================================================
# RV placeholder (currently default 0 and excluded from CapNav)
# =========================================================
def rv_applicable_from_raw(raw_output: str) -> bool:
    """
    RV (per paper) applies to infeasible predictions (y_hat=0) where
    the model returns a proposed route AND a short rationale.
    Since you currently set RV=0 by default, we keep a lightweight
    applicability check for future integration.

    You can make this stricter later once you standardize the output format.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return False
    # Heuristic: look for "Fail:" or "Reason:" as rationale marker
    return bool(re.search(r"\|\s*(fail|reason)\s*:", raw_output, flags=re.IGNORECASE))

# =========================================================
# Score one record
# =========================================================
def score_record(rec: Dict) -> Dict:
    qid = rec.get("qid")
    scene = rec.get("scene")
    agent = rec.get("agent")
    question = rec.get("question")
    output = rec.get("output", "")
    entry_idx = rec.get("entry_idx")
    status = rec.get("status")

    base = {
        "qid": qid,
        "scene": scene,
        "agent": agent,
        "entry_idx": entry_idx,
        "status": status,
        "question": question,
    }

    if not scene or not agent or not question:
        return {**base, "ok": False, "error": "missing_scene_agent_or_question"}

    graph, traverse = load_graph_traverse(scene)
    name_to_id, adj, edge_set = build_graph_index(graph)
    node_names = list(name_to_id.keys())

    st = derive_src_tgt_by_reverse_match(question, name_to_id)
    if not st["ok"]:
        return {**base, "ok": False, "error": st["error"], "derive_method": st.get("method")}

    src_id, tgt_id = st["src_id"], st["tgt_id"]

    # Compute GT doable regardless of output format
    gt_doable = exists_traversable_path(adj, traverse, agent, src_id, tgt_id)

    parsed = parse_vlm_output_text(output)
    if not parsed["ok"]:
        # parse failure affects ONLY binary (F1 etc.), not PV/RTA/RV aggregates
        binary_bucket = "FN" if gt_doable else "FP"
        return {
            **base,
            "ok": True,
            "format_error": True,
            "format_error_type": parsed["error"],
            "derive_method": st.get("method"),
            "src_name": st.get("src_name"),
            "tgt_name": st.get("tgt_name"),
            "src_id": src_id,
            "tgt_id": tgt_id,
            "vlm_answer": "fail",
            "vlm_path_names_raw": [],
            "vlm_path_names_mapped": [],
            "vlm_path_ids": [],
            "pv_valid": None,                 # not counted
            "pv_errors": [],                  # not counted
            "rta_applicable": False,          # not counted
            "traversability_score": None,     # not counted
            "rv_applicable": False,           # not counted
            "rv_score": None,                 # not counted
            "ground_truth_doable": gt_doable,
            "correct_answer": False,
            "binary_bucket": binary_bucket,
        }

    pred_ids: List[Optional[str]] = []
    pred_mapped_names: List[Optional[str]] = []
    for pn in parsed["path_names"]:
        bn = best_match_pred_name(pn, node_names)
        pred_mapped_names.append(bn)
        pred_ids.append(name_to_id[bn] if bn else None)

    pv_valid, pv_errors = pv_strict(pred_ids, src_id, tgt_id, edge_set)

    # RTA per paper: conditioned on y_hat = 1 AND a valid P_hat
    rta_applicable = (parsed["answer"] == "yes" and pv_valid)
    trav = None
    if rta_applicable:
        trav = traversability_score(traverse, agent, [x for x in pred_ids if x is not None])

    # RV per paper: conditioned on y_hat = 0 (negative predictions)
    # You said: currently default is 0, and excluded from CapNav composite.
    # We therefore set rv_score=0.0 when applicable, else None (not counted).
    rv_applicable = (parsed["answer"] == "no")
    # Optional: stricter applicability requiring rationale marker in raw output
    # rv_applicable = (parsed["answer"] == "no" and rv_applicable_from_raw(parsed["raw"]))
    rv_score = 0.0 if rv_applicable else None

    correct_answer = ((parsed["answer"] == "yes" and gt_doable) or (parsed["answer"] == "no" and not gt_doable))

    return {
        **base,
        "ok": True,
        "format_error": False,
        "format_error_type": None,
        "binary_bucket": None,  # only set for parse failure branch
        "derive_method": st.get("method"),
        "src_name": st.get("src_name"),
        "tgt_name": st.get("tgt_name"),
        "src_id": src_id,
        "tgt_id": tgt_id,
        "vlm_answer": parsed["answer"],
        "vlm_path_names_raw": parsed["path_names"],
        "vlm_path_names_mapped": pred_mapped_names,
        "vlm_path_ids": pred_ids,
        "pv_valid": pv_valid,
        "pv_errors": pv_errors,
        "rta_applicable": rta_applicable,
        "traversability_score": trav,
        "rv_applicable": rv_applicable,
        "rv_score": rv_score,
        "ground_truth_doable": gt_doable,
        "correct_answer": correct_answer,
    }

# =========================================================
# Main
# =========================================================
def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"RESULTS_DIR not found: {RESULTS_DIR}")
    if not GT_ROOT.exists():
        raise FileNotFoundError(f"GT_ROOT not found: {GT_ROOT}")

    input_files = choose_input_files()
    if not input_files:
        raise FileNotFoundError(f"No input jsonl found. Expected {DEFAULT_JSONL} or *.jsonl under {RESULTS_DIR}")

    print("Scoring input files:")
    for p in input_files:
        print(f"  - {p}")

    if OUT_PER_RECORD.exists():
        OUT_PER_RECORD.unlink()

    agg = {
        "n_records": 0,
        "n_ok": 0,
        "n_fail": 0,      # structural failure only
        "n_correct": 0,

        # PV: only parse-success subset
        "pv_total": 0,
        "pv_valid": 0,

        # RTA: only y_hat=1 and PV valid subset
        "rta_count": 0,
        "rta_sum": 0.0,

        # RV: only y_hat=0 subset (currently 0.0 default)
        "rv_count": 0,
        "rv_sum": 0.0,

        "format_errors": 0,
    }
    fail_reason_count = defaultdict(int)
    derive_method_count = defaultdict(int)

    # confusion matrix for binary feasibility (includes parse failures)
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    format_error_details = defaultdict(int)

    # coverage bookkeeping (optional but useful for interpretation)
    pred_yes_count = 0
    pred_no_count = 0

    with OUT_PER_RECORD.open("w", encoding="utf-8") as fout:
        for fpath in input_files:
            for rec in iter_jsonl(fpath):
                agg["n_records"] += 1
                scored = score_record(rec)
                fout.write(json.dumps(scored, ensure_ascii=False) + "\n")

                if not scored.get("ok", False):
                    agg["n_fail"] += 1
                    fail_reason_count[scored.get("error", "unknown")] += 1
                    continue

                agg["n_ok"] += 1
                derive_method_count[scored.get("derive_method", "unknown")] += 1

                if scored.get("correct_answer", False):
                    agg["n_correct"] += 1

                if scored.get("format_error", False):
                    agg["format_errors"] += 1
                    format_error_details[scored.get("format_error_type", "unknown")] += 1

                # update confusion matrix (including parse failures)
                bb = scored.get("binary_bucket")
                if bb in cm:
                    cm[bb] += 1
                else:
                    pred_yes = (scored.get("vlm_answer") == "yes")
                    pred_no = (scored.get("vlm_answer") == "no")
                    if pred_yes:
                        pred_yes_count += 1
                    if pred_no:
                        pred_no_count += 1

                    gt = bool(scored.get("ground_truth_doable"))
                    if gt and pred_yes:
                        cm["TP"] += 1
                    elif (not gt) and pred_yes:
                        cm["FP"] += 1
                    elif gt and (not pred_yes):
                        cm["FN"] += 1
                    else:
                        cm["TN"] += 1

                # PV statistics: ONLY when parse succeeded
                if scored.get("format_error", False) is False:
                    agg["pv_total"] += 1
                    if scored.get("pv_valid", False):
                        agg["pv_valid"] += 1

                # RTA statistics: ONLY when applicable
                if scored.get("traversability_score") is not None:
                    agg["rta_count"] += 1
                    agg["rta_sum"] += float(scored["traversability_score"])

                # RV statistics: ONLY when applicable (y_hat=0). Currently always 0.0 then.
                if scored.get("rv_score") is not None:
                    agg["rv_count"] += 1
                    agg["rv_sum"] += float(scored["rv_score"])

    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    bin_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    pv_rate = agg["pv_valid"] / agg["pv_total"] if agg["pv_total"] else 0.0
    rta_mean = agg["rta_sum"] / agg["rta_count"] if agg["rta_count"] else 0.0
    rv_mean = agg["rv_sum"] / agg["rv_count"] if agg["rv_count"] else 0.0

    # Composite CapNav score (your current setting): average of F1, PV, RTA only.
    capnav_score = (f1 + pv_rate + rta_mean) / 3.0

    summary = {
        **agg,
        "accuracy": agg["n_correct"] / agg["n_ok"] if agg["n_ok"] else 0.0,

        # PV and RTA as defined above
        "pv_validity_rate": pv_rate,
        "avg_traversability_score": rta_mean,

        # RV placeholder (reported, but excluded from capnav_score below)
        "avg_rv_score": rv_mean,

        # Composite (ONLY F1, PV, RTA)
        "capnav_score": capnav_score,
        "capnav_components": {
            "f1": f1,
            "pv": pv_rate,
            "rta": rta_mean,
            "rv_excluded_default": True,
        },

        # coverages (helpful for writing results)
        "coverage": {
            "pv_applicable_frac_over_ok": agg["pv_total"] / agg["n_ok"] if agg["n_ok"] else 0.0,
            "rta_applicable_frac_over_ok": agg["rta_count"] / agg["n_ok"] if agg["n_ok"] else 0.0,
            "rv_applicable_frac_over_ok": agg["rv_count"] / agg["n_ok"] if agg["n_ok"] else 0.0,
        },

        "fail_reason_count": dict(fail_reason_count),
        "derive_method_count": dict(derive_method_count),
        "format_error_details": dict(format_error_details),

        "binary_classification": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": bin_acc,
        },

        "input_files": [str(p) for p in input_files],
        "per_record_output": str(OUT_PER_RECORD),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n  {OUT_PER_RECORD}\n  {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
