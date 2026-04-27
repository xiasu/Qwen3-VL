#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for Qwen-VL models using TRL's SFTTrainer.

pip install math_verify

Example usage:
    python qwenvl/train/train_qwen_trl_grpo.py \
        --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
        --output_dir ./output/qwen-vl-8b-trl \
        --per_device_train_batch_size 2 \
        --per_device_train_batch_size 8 \
        --max_completion_length 1024 \
        --num_generations 8 \
        --max_prompt_length 2048 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --bf16 True \
        --gradient_checkpointing True \
        --use_peft True \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules "q_proj", "v_proj" \
        --log_completions
"""

from dataclasses import dataclass, field
from typing import Optional

import re
import json
import string
import sys
import torch
from pathlib import Path
from collections import deque
from difflib import SequenceMatcher
from datasets import load_dataset
from transformers import AutoProcessor
from typing import Optional, Dict, List, Tuple

# Add project root to Python path (same as train_qwen.py)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trl import (
    ModelConfig,
    ScriptArguments,
    GRPOConfig,
    GRPOTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from qwenvl.train.argument import DataArguments
from qwenvl.data.data_processor import make_supervised_data_module
from torch.utils.data import Dataset


@dataclass
class QwenScriptArguments(ScriptArguments):
    """Extended script arguments for Qwen-VL training."""
    
    # Note: max_pixels, min_pixels, and video_fps are defined in DataArguments
    # to avoid argparse conflicts. Use --max_pixels, --min_pixels, --video_fps
    # which will be parsed into DataArguments.

    data_root: Optional[str] = field(
        default="", metadata={"help": "Root directory for image/video files"}
    )
    lora_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to previous LoRA checkpoint to load before GRPO training"}
    )


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

# =========================================================
# CapNav reward function - helper functions from score.py
# =========================================================

# Ground truth root directory (adjust if needed)
CAPNAV_GT_ROOT = Path("/mmfs1/gscratch/makelab/ruiqi/datasets/capnav/ground_truth")
_SCENE_CACHE: Dict[str, Tuple[Dict, Dict]] = {}

def norm_text(s: str) -> str:
    """Normalize text for matching."""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

def parse_vlm_output_text(output: str) -> Dict:
    """Parse VLM output to extract answer and path."""
    if not isinstance(output, str) or not output.strip():
        return {"ok": False, "error": "empty_output", "raw": output}

    # Remove reasoning tags if present (matches format_reward pattern)
    cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    
    # Extract content from <answer> tags if present
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        cleaned = answer_match.group(1).strip()

    # Accept: Answer: (A|B) True|False | Path: ...
    m = re.search(
        r"Answer:\s*\((A|B)\)\s*(True|False)\s*\|\s*Path:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE
    )
    if not m:
        return {"ok": False, "error": "cannot_parse_answer_path", "raw": cleaned}

    answer = "yes" if m.group(2).lower() == "true" else "no"
    path_str = m.group(3).strip()

    # Cut off extra fields after path
    path_str = re.split(r"\s*\|\s*(?:Fail|Reason)\s*:", path_str, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    path_names = [p.strip() for p in re.split(r"\s*(?:->|→)\s*", path_str) if p.strip()]

    if answer == "yes" and len(path_names) < 2:
        return {"ok": False, "error": "yes_without_path", "raw": cleaned}

    return {"ok": True, "answer": answer, "path_names": path_names, "raw": cleaned}

def find_from_to_indices(qn: str) -> Optional[Tuple[int, int]]:
    """Find 'from' and 'to' indices in question."""
    m_from = re.search(r"\bfrom\b", qn)
    if not m_from:
        return None
    m_to = re.search(r"\bto\b", qn[m_from.end():])
    if not m_to:
        return None
    to_start = m_from.end() + m_to.start()
    return (m_from.end(), to_start)

def all_node_hits_in_question(qn: str, node_names: List[str]) -> List[Dict]:
    """Find all node name hits in question."""
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
    """Derive source and target from question by matching node names."""
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

def load_graph_traverse(scene: str) -> Tuple[Dict, Dict]:
    """Load graph and traverse data for a scene."""
    if scene in _SCENE_CACHE:
        return _SCENE_CACHE[scene]

    graph_path = CAPNAV_GT_ROOT / "graphs" / f"{scene}-graph.json"
    traverse_path = CAPNAV_GT_ROOT / "traverse" / f"{scene}-traverse.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph: {graph_path}")
    if not traverse_path.exists():
        raise FileNotFoundError(f"Missing traverse: {traverse_path}")

    graph = json.loads(graph_path.read_text())
    traverse = json.loads(traverse_path.read_text())
    _SCENE_CACHE[scene] = (graph, traverse)
    return graph, traverse

def build_graph_index(graph: Dict) -> Tuple[Dict[str, str], Dict[str, List[str]], set]:
    """Build graph index from graph data."""
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

def agent_key(traverse: Dict, agent: str) -> Optional[str]:
    """Get agent key from traverse dict."""
    a = agent.upper()
    if a in traverse:
        return a
    if a == "HUMANOID" and "ROBOT" in traverse:
        return "ROBOT"
    return None

def edge_traversable(traverse: Dict, agent: str, u: str, v: str) -> bool:
    """Check if edge is traversable for agent."""
    k = agent_key(traverse, agent)
    if not k:
        return False
    ad = traverse[k]
    return bool(ad.get(f"{u}|{v}", ad.get(f"{v}|{u}", {})).get("traversable", False))

def traversability_score(traverse: Dict, agent: str, path_ids: List[str]) -> Optional[float]:
    """Calculate traversability score for a path."""
    if not path_ids or len(path_ids) < 2:
        return None
    total = len(path_ids) - 1
    ok = 0
    for i in range(total):
        if edge_traversable(traverse, agent, path_ids[i], path_ids[i + 1]):
            ok += 1
    return ok / total

def pv_strict(path_ids: List[Optional[str]], src_id: str, tgt_id: str, edge_set: set) -> Tuple[bool, List[str]]:
    """Check path validity strictly."""
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

def exists_traversable_path(adj: Dict[str, List[str]], traverse: Dict, agent: str, src: str, tgt: str) -> bool:
    """Check if traversable path exists using BFS."""
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

def best_match_pred_name(pred_name: str, node_names: List[str]) -> Optional[str]:
    """Find best matching node name for predicted name."""
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

def capnav_reward(completions, **kwargs):
    """
    CapNav reward function that computes composite score based on:
    - F1 score (binary classification correctness)
    - PV (Path Validity) - whether predicted path is valid
    - RTA (Route Traversability Accuracy) - traversability of valid paths
    
    Component weights (adjustable):
    - WEIGHT_F1: weight for F1 score component
    - WEIGHT_PV: weight for Path Validity component  
    - WEIGHT_RTA: weight for Route Traversability Accuracy component
    
    Final reward = (WEIGHT_F1 * f1_score + WEIGHT_PV * pv_score + WEIGHT_RTA * rta_score) / (WEIGHT_F1 + WEIGHT_PV + WEIGHT_RTA)
    
    Args:
        completions: List of model completion strings
        **kwargs: Should contain dataset batch with 'scene', 'agent', 'question' fields
        
    Returns:
        List of reward values (one per completion)
    """
    # =========================================================
    # CONFIGURABLE WEIGHTS - Adjust these to change component importance
    # =========================================================
    WEIGHT_F1 = 1.0      # Weight for binary classification F1 score
    WEIGHT_PV = 1.0      # Weight for Path Validity score
    WEIGHT_RTA = 1.0     # Weight for Route Traversability Accuracy score
    # Default: equal weights (1/3 each in final average)
    # =========================================================
    
    rewards = []
    
    # Extract dataset fields from kwargs
    # GRPOTrainer typically passes dataset batch through kwargs
    # Try common field names
    scenes = kwargs.get("scene", kwargs.get("scenes", []))
    agents = kwargs.get("agent", kwargs.get("agents", []))
    questions = kwargs.get("question", kwargs.get("questions", []))
    
    # If not found, try to get from batch (might be in a different format)
    if not scenes and "batch" in kwargs:
        batch = kwargs["batch"]
        scenes = batch.get("scene", [])
        agents = batch.get("agent", [])
        questions = batch.get("question", [])
    
    # Handle single value vs list
    if isinstance(scenes, str):
        scenes = [scenes] * len(completions)
    if isinstance(agents, str):
        agents = [agents] * len(completions)
    if isinstance(questions, str):
        questions = [questions] * len(completions)
    
    # Ensure we have the right number of items
    n = len(completions)
    if len(scenes) != n:
        scenes = [scenes[0]] * n if scenes else [None] * n
    if len(agents) != n:
        agents = [agents[0]] * n if agents else [None] * n
    if len(questions) != n:
        questions = [questions[0]] * n if questions else [None] * n
    
    for i, completion in enumerate(completions):
        scene = scenes[i] if i < len(scenes) else None
        agent = agents[i] if i < len(agents) else None
        question = questions[i] if i < len(questions) else None
        
        # If missing required fields, return zero reward
        if not scene or not agent or not question:
            rewards.append(0.0)
            continue
        
        try:
            # Load graph and traverse data
            graph, traverse = load_graph_traverse(scene)
            name_to_id, adj, edge_set = build_graph_index(graph)
            node_names = list(name_to_id.keys())
            
            # Derive source and target from question
            st = derive_src_tgt_by_reverse_match(question, name_to_id)
            if not st["ok"]:
                rewards.append(0.0)
                continue
            
            src_id, tgt_id = st["src_id"], st["tgt_id"]
            
            # Compute ground truth doable
            gt_doable = exists_traversable_path(adj, traverse, agent, src_id, tgt_id)
            
            # Parse completion
            parsed = parse_vlm_output_text(completion)
            
            # =========================================================
            # Component 1: F1 Score (Binary Classification)
            # =========================================================
            if not parsed["ok"]:
                # Parse failure: treat as wrong prediction
                f1_score = 0.0
                pv_score = 0.0
                rta_score = 0.0
            else:
                # Check binary correctness
                pred_yes = (parsed["answer"] == "yes")
                correct_answer = ((pred_yes and gt_doable) or (not pred_yes and not gt_doable))
                
                # For F1, we need to compute precision/recall across batch
                # Here we compute per-sample correctness as a proxy
                # True positive: pred_yes and gt_doable
                # True negative: not pred_yes and not gt_doable
                # False positive: pred_yes and not gt_doable
                # False negative: not pred_yes and gt_doable
                
                # For individual reward, we use correctness as binary score
                # (In aggregate, this would give F1)
                f1_score = 1.0 if correct_answer else 0.0
            
            # =========================================================
            # Component 2: Path Validity (PV)
            # =========================================================
            pred_ids: List[Optional[str]] = []
            if not parsed["ok"]:
                pv_score = 0.0
            else:
                # Map predicted path names to node IDs
                for pn in parsed["path_names"]:
                    bn = best_match_pred_name(pn, node_names)
                    pred_ids.append(name_to_id[bn] if bn else None)
                
                # Check path validity
                pv_valid, pv_errors = pv_strict(pred_ids, src_id, tgt_id, edge_set)
                pv_score = 1.0 if pv_valid else 0.0
            
            # =========================================================
            # Component 3: Route Traversability Accuracy (RTA)
            # =========================================================
            if not parsed["ok"]:
                rta_score = 0.0
            else:
                # RTA applies only when answer is "yes" and path is valid
                rta_applicable = (parsed["answer"] == "yes" and pv_score > 0.5)
                
                if rta_applicable:
                    # Reuse pred_ids from PV calculation
                    clean_path_ids = [x for x in pred_ids if x is not None]
                    trav = traversability_score(traverse, agent, clean_path_ids)
                    rta_score = trav if trav is not None else 0.0
                else:
                    rta_score = 0.0
            
            # =========================================================
            # Composite Reward Calculation
            # =========================================================
            # Normalize weights
            total_weight = WEIGHT_F1 + WEIGHT_PV + WEIGHT_RTA
            if total_weight == 0:
                total_weight = 1.0
            
            composite_reward = (
                WEIGHT_F1 * f1_score +
                WEIGHT_PV * pv_score +
                WEIGHT_RTA * rta_score
            ) / total_weight
            
            rewards.append(float(composite_reward))
            
        except Exception as e:
            # On any error, return zero reward
            print(f"Error computing capnav reward for completion {i}: {e}")
            rewards.append(0.0)
    
    return rewards

class MetadataPreservingDataset(Dataset):
    """Wrapper dataset that preserves metadata fields for reward functions and adds prompt field for GRPO."""
    
    def __init__(self, base_dataset, processor, metadata_fields=None):
        self.base_dataset = base_dataset
        # Use processor from base_dataset if available (has the right config), otherwise use passed one
        if hasattr(base_dataset, 'processor'):
            self.processor = base_dataset.processor
        else:
            self.processor = processor
        self.metadata_fields = metadata_fields or []
        # Store metadata from original data if available
        if hasattr(base_dataset, 'list_data_dict'):
            self.metadata_list = base_dataset.list_data_dict
        else:
            self.metadata_list = None
        # Store data_args if available
        if hasattr(base_dataset, 'data_args'):
            self.data_args = base_dataset.data_args
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _extract_metadata(self, metadata_dict, field):
        """Extract metadata field from various possible locations."""
        # Try direct field first
        if field in metadata_dict:
            return metadata_dict[field]
        
        # For capnav, try to extract from video name or conversations
        if field == "scene":
            # Try to extract scene from video path/name
            video = metadata_dict.get("video", "")
            if video:
                # Scene might be in video name (e.g., "HM3D00003_video.mp4")
                import re
                match = re.search(r"(HM3D\d+|MP3D\d+)", video)
                if match:
                    return match.group(1)
        
        if field == "agent" or field == "question":
            # Try to extract from conversations
            conversations = metadata_dict.get("conversations", [])
            for conv in conversations:
                if isinstance(conv, dict) and conv.get("from") == "human":
                    value = conv.get("value", "")
                    if field == "agent":
                        # Extract agent from question text (e.g., "Can [AGENT] move...")
                        import re
                        match = re.search(r"\[([A-Z]+)\]", value)
                        if match:
                            return match.group(1)
                    elif field == "question":
                        # The question is the human's value
                        return value
        
        # Check if field is directly in metadata
        if field in metadata_dict:
            return metadata_dict[field]
        
        return None
    
    def _build_messages_for_prompt(self, item: Dict) -> List[Dict]:
        """Build messages from conversations for prompt generation (same logic as data_processor._build_messages)."""
        from pathlib import Path
        
        # Extract and normalize images and videos
        images = item.get("image") or []
        if isinstance(images, str):
            images = [images]

        videos = item.get("video") or []
        if isinstance(videos, str):
            videos = [videos]

        # Get base path and resolve media paths
        data_path = item.get("data_path", "")
        base_path = Path(data_path) if data_path else None
        
        # Build media pools with absolute paths
        def resolve_media_path(media_path):
            if not media_path:
                return ""
            media_path_obj = Path(media_path)
            if media_path_obj.is_absolute():
                return str(media_path_obj)
            elif base_path:
                return str((base_path / media_path).resolve())
            else:
                return str(media_path)
        
        image_pool = [
            {"type": "image", "image": resolve_media_path(img)} for img in images
        ]
        video_pool = [
            {"type": "video", "video": resolve_media_path(vid)} for vid in videos
        ]

        messages = []
        for turn in item.get("conversations", []):
            role = "user" if turn.get("from") == "human" else "assistant"
            text: str = turn.get("value", "")

            if role == "user":
                content = []
                # Split text by <image> or <video> placeholders while keeping delimiters
                text_parts = re.split(r"(<image>|<video>)", text)

                for seg in text_parts:
                    if seg == "<image>":
                        if not image_pool:
                            raise ValueError(
                                "Number of <image> placeholders exceeds the number of provided images"
                            )
                        content.append(image_pool.pop(0))
                    elif seg == "<video>":
                        if not video_pool:
                            raise ValueError(
                                "Number of <video> placeholders exceeds the number of provided videos"
                            )
                        content.append(video_pool.pop(0))
                    elif seg.strip():
                        content.append({"type": "text", "text": seg.strip()})

                messages.append({"role": role, "content": content})
            else:
                # Assistant messages contain only text
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        return messages
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Always try to add prompt field - GRPOTrainer requires it
        prompt_added = False
        
        # Add metadata fields and prompt if available
        if self.metadata_list and idx < len(self.metadata_list):
            metadata = self.metadata_list[idx]
            # Extract metadata fields (scene, agent, question) from original data
            if isinstance(metadata, dict):
                for field in self.metadata_fields:
                    value = self._extract_metadata(metadata, field)
                    if value is not None:
                        # Store as string to ensure it's serializable
                        item[field] = str(value) if not isinstance(value, str) else value
                
                # Build prompt from conversations for GRPOTrainer
                # GRPOTrainer needs a "prompt" field for generation
                try:
                    messages = self._build_messages_for_prompt(metadata)
                    # Apply chat template to create the prompt string
                    # GRPOTrainer will tokenize this internally
                    prompt_text = self.processor.apply_chat_template(
                        messages, 
                        add_generation_prompt=True,
                        tokenize=False  # Get text string, not tokens
                    )
                    # Ensure it's a string
                    if isinstance(prompt_text, list):
                        # If it returns a list (some processors do), join it
                        prompt_text = "".join(prompt_text) if all(isinstance(x, str) for x in prompt_text) else str(prompt_text)
                    item["prompt"] = str(prompt_text)
                    prompt_added = True
                except Exception as e:
                    # If prompt generation fails, try to decode from input_ids as fallback
                    print(f"Warning: Failed to build prompt from conversations for item {idx}: {e}")
        
        # Fallback: if we couldn't build prompt from conversations, try to extract from input_ids
        if not prompt_added and "input_ids" in item:
            try:
                # Decode the input_ids to get the prompt text
                input_ids = item["input_ids"]
                if isinstance(input_ids, list) and len(input_ids) > 0:
                    input_ids = input_ids[0]
                elif hasattr(input_ids, 'squeeze'):
                    input_ids = input_ids.squeeze(0)
                
                # Decode to get the full text
                full_text = self.processor.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False
                )
                # For GRPO, we need just the prompt part (before the answer)
                # The answer typically starts after certain tokens - we'll use the full decoded text
                # and let GRPOTrainer handle the splitting
                item["prompt"] = full_text
                prompt_added = True
            except Exception as e2:
                print(f"Warning: Fallback prompt generation from input_ids failed for item {idx}: {e2}")
        
        # Last resort: if we still don't have a prompt, create a minimal one
        if not prompt_added:
            print(f"Warning: Could not generate prompt for item {idx}, using empty string")
            item["prompt"] = ""
        
        return item


if __name__ == "__main__":
    # Parse arguments - add DataArguments for dataset loading
    parser = TrlParser((QwenScriptArguments, GRPOConfig, ModelConfig, DataArguments))
    script_args, training_args, model_args, data_args = parser.parse_args_and_config()

    # Set training configurations
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_text_field = ""  # We'll use formatting function
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model & Processor
    ################
    # Determine dtype: check training_args.bf16 first (like train_qwen.py), then model_args.dtype
    if training_args.bf16:
        dtype = torch.bfloat16
    elif model_args.dtype and model_args.dtype not in ["auto", None]:
        dtype = getattr(torch, model_args.dtype)
    else:
        dtype = None

    # Determine attention implementation
    # If not specified, default to flash_attention_2 (same as train_qwen.py)
    attn_implementation = model_args.attn_implementation or "flash_attention_2"
    
    # Verify flash_attention_2 is available if requested
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn
            print("✓ flash_attention_2 is available and will be used")
        except ImportError:
            print("⚠ Warning: flash_attention_2 requested but flash_attn package not found")
            print("  Falling back to sdpa (scaled dot product attention)")
            print("  Install flash_attn for better performance: pip install flash-attn")
            attn_implementation = "sdpa"

    # Determine model class based on model name
    # Note: Qwen3-VL-30B-A3B uses MoE architecture (A3B = Active 3B parameters)
    if "qwen3" in model_args.model_name_or_path.lower() and (
        "moe" in model_args.model_name_or_path.lower()
        or "a3b" in model_args.model_name_or_path.lower()
    ):
        from transformers import Qwen3VLMoeForConditionalGeneration

        model_class = Qwen3VLMoeForConditionalGeneration
    elif "qwen3" in model_args.model_name_or_path.lower():
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration

        model_class = Qwen2VLForConditionalGeneration

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    
    # Print dtype and attention implementation info
    print(f"Model dtype: {dtype}")
    print(f"Attention implementation: {attn_implementation}")

    # Add quantization config if specified
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Load model
    print(f"Loading model: {model_args.model_name_or_path}")
    print(f"Model class: {model_class.__name__}")
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    # Enable MoE auxiliary loss for MoE models
    # This is required for TRL to include the load balancing/auxiliary loss
    is_moe_model = "moe" in model_class.__name__.lower()
    if is_moe_model:
        model.config.output_router_logits = True
        print("✓ Enabled MoE auxiliary loss (output_router_logits=True)")
        print(
            "  TRL will automatically include router load balancing loss in training"
        )

    # Load LoRA checkpoint if provided
    # Note: This should be done BEFORE creating peft_config, as loading a checkpoint
    # will already set up the PEFT model structure
    if script_args.lora_checkpoint_path:
        from peft import PeftModel
        from pathlib import Path
        
        lora_path = Path(script_args.lora_checkpoint_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        
        print(f"Loading LoRA checkpoint from: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path),is_trainable=True)
        print("✓ LoRA checkpoint loaded successfully")
        print("  Note: GRPOTrainer will continue training from this checkpoint")
        print("  The model is already a PEFT model, so peft_config will be loaded from checkpoint")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set processor pixels from data_args (these are now only in DataArguments)
    if hasattr(processor, "image_processor"):
        if hasattr(data_args, "max_pixels") and data_args.max_pixels:
            processor.image_processor.max_pixels = data_args.max_pixels
        if hasattr(data_args, "min_pixels") and data_args.min_pixels:
            processor.image_processor.min_pixels = data_args.min_pixels
    
    if hasattr(processor, "video_processor"):
        if hasattr(data_args, "video_fps") and data_args.video_fps:
            processor.video_processor.fps = data_args.video_fps

    ################
    # Dataset
    ################
    # Use DataArguments to load dataset the same way as train_qwen.py
    if not data_args.dataset_use:
        # Fallback to script_args if dataset_use not provided
        if hasattr(script_args, "dataset_name") and script_args.dataset_name:
            data_args.dataset_use = script_args.dataset_name
        else:
            raise ValueError("Either --dataset_use (DataArguments) or --dataset_name (ScriptArguments) must be provided")
    
    print(f"Loading dataset: {data_args.dataset_use}")
    
    # Set model_type for data processing
    if "qwen3" in model_args.model_name_or_path.lower() and (
        "moe" in model_args.model_name_or_path.lower()
        or "a3b" in model_args.model_name_or_path.lower()
    ):
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        data_args.model_type = "qwen2.5vl"
    else:
        data_args.model_type = "qwen2vl"
    
    # Load dataset using make_supervised_data_module (same as train_qwen.py)
    data_module = make_supervised_data_module(processor, data_args=data_args)
    base_train_dataset = data_module["train_dataset"]
    base_eval_dataset = data_module.get("eval_dataset")
    
    # Wrap dataset to preserve metadata fields for capnav_reward and add prompt field for GRPO
    # For capnav dataset, we need to preserve: scene, agent, question
    metadata_fields = ["scene", "agent", "question", "qid"]
    train_dataset = MetadataPreservingDataset(base_train_dataset, processor, metadata_fields=metadata_fields)
    eval_dataset = (
        MetadataPreservingDataset(base_eval_dataset, processor, metadata_fields=metadata_fields)
        if base_eval_dataset is not None and training_args.eval_strategy != "no"
        else None
    )

    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Calculate and print expected training steps
    num_gpus = training_args.world_size if hasattr(training_args, 'world_size') and training_args.world_size else 1
    if not num_gpus or num_gpus == 0:
        # Fallback: try to get from environment or default to NPROC_PER_NODE
        import os
        num_gpus = int(os.environ.get('WORLD_SIZE', os.environ.get('NPROC_PER_NODE', '1')))
    
    effective_batch_size = (
        training_args.per_device_train_batch_size * 
        num_gpus * 
        training_args.gradient_accumulation_steps
    )
    steps_per_epoch = len(train_dataset) // effective_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print("\n" + "="*60)
    print("Training Configuration Summary:")
    print("="*60)
    print(f"Dataset size: {len(train_dataset):,} samples")
    print(f"Per device batch size: {training_args.per_device_train_batch_size}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {int(total_steps * training_args.warmup_ratio) if hasattr(training_args, 'warmup_ratio') else 'N/A'}")
    print("="*60 + "\n")

    ################
    # PEFT Config
    ################
    # If we loaded a LoRA checkpoint, the model is already a PEFT model
    # In this case, we should not create a new peft_config, but use None
    # The GRPOTrainer will use the existing PEFT structure
    if script_args.lora_checkpoint_path:
        print("Using existing PEFT model from loaded checkpoint")
        peft_config = None  # Model already has PEFT structure
    else:
        peft_config = get_peft_config(model_args)
        if peft_config:
            print(f"Using PEFT with config: {peft_config}")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[capnav_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    
    # Print GRPO-specific configuration and check actual step count
    print("\n" + "="*60)
    print("GRPO Configuration:")
    print("="*60)
    num_gens = getattr(training_args, 'num_generations', None)
    if num_gens:
        print(f"Number of generations per prompt: {num_gens}")
        print(f"⚠ IMPORTANT: GRPO generates {num_gens} completions per sample")
        print(f"  This means each sample is processed {num_gens} times during training")
        print(f"  Standard calculation: {total_steps} steps")
        print(f"  GRPO effective steps (if per-generation): {total_steps * num_gens}")
        print(f"  Your observed steps: 5,242")
        print(f"  Ratio: {5242 / total_steps:.2f}x (close to {num_gens}x)")
    
    # Check what the trainer actually calculates
    # GRPOTrainer might use a different step calculation
    try:
        # Access the trainer's internal step calculation
        if hasattr(trainer, 'get_train_dataloader'):
            dataloader = trainer.get_train_dataloader()
            if hasattr(dataloader, '__len__'):
                dataloader_len = len(dataloader)
                print(f"\nActual dataloader length: {dataloader_len}")
                if dataloader_len != steps_per_epoch:
                    print(f"⚠ Mismatch! Expected {steps_per_epoch}, got {dataloader_len}")
    except Exception as e:
        print(f"Could not check dataloader length: {e}")
    
    # Check training args for max_steps override
    if hasattr(training_args, 'max_steps') and training_args.max_steps:
        print(f"max_steps is set to: {training_args.max_steps}")
    print("="*60 + "\n")

    # Check the actual step count that will be used
    # GRPO might calculate steps differently
    try:
        # Try to access the trainer's step calculation
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'max_steps'):
            actual_max_steps = trainer.state.max_steps
            print(f"\n⚠ Trainer calculated max_steps: {actual_max_steps:,}")
            if actual_max_steps and actual_max_steps != total_steps:
                print(f"  This differs from our calculation: {total_steps:,}")
                print(f"  Difference: {actual_max_steps - total_steps:,} steps")
                if num_gens and abs(actual_max_steps / total_steps - num_gens) < 0.1:
                    print(f"  ✓ This matches num_generations={num_gens} multiplier")
                    print(f"  GRPO appears to process each sample {num_gens} times")
    except Exception as e:
        print(f"Could not check trainer max_steps: {e}")
    
    print("\nStarting training...")
    trainer.train()

    # Save model and processor
    # Note: When using PEFT (LoRA), save_model will automatically save only the adapter weights
    # The output will contain the LoRA weights that can be loaded with PeftModel.from_pretrained
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    
    # If using PEFT, the saved model will be in adapter format
    # To use it later, load with: PeftModel.from_pretrained(base_model, output_dir)
    if peft_config is not None or script_args.lora_checkpoint_path:
        print("✓ LoRA adapter weights saved (PEFT format)")
        print(f"  To load later: PeftModel.from_pretrained(base_model, '{training_args.output_dir}')")

    if training_args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Training completed!")