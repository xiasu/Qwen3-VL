import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


# =========================
# Paths inferred from script location
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent     # /.../Qwen3-VL
USER_ROOT = PROJECT_ROOT.parent                   # /.../ruiqi

ALL_SCENES_JSON = PROJECT_ROOT / "all_scenes.json"
OUTPUT_DIR = PROJECT_ROOT / "capnav_qa_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_PATH = PROJECT_ROOT / "output" / "capnav-test-1" / "checkpoint-843"
VIDEO_ROOT = USER_ROOT / "datasets" / "capnav" / "videos"

# Output files (global across ALL questions)
META_JSON = OUTPUT_DIR / "all.meta.json"
RESULTS_JSONL = OUTPUT_DIR / "all.results.jsonl"
FAILS_JSONL = OUTPUT_DIR / "all.failures.jsonl"
PROGRESS_JSON = OUTPUT_DIR / "all.progress.json"

# Model config (keep identical to your code)
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
PROCESSOR_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
MAX_NEW_TOKENS = 9280


# =========================
# Helpers (behavior unchanged)
# =========================
def strip_video_marker(text: str) -> str:
    return re.sub(r"^\s*<video>\s*\n?", "", (text or "").strip(), flags=re.IGNORECASE)

def parse_question(prompt_text: str) -> str:
    m = re.search(r"\[Question\]\s*\n(.+?)(?:\n\s*\n|\n\[|$)", prompt_text, flags=re.DOTALL)
    if not m:
        return ""
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    return lines[0] if lines else ""

def parse_agent(question_text: str) -> str:
    m = re.search(r"\[([A-Za-z0-9_]+)\]", question_text)
    return m.group(1) if m else ""

def make_qid(scene_id: str, agent: str, question_text: str) -> str:
    raw = f"{scene_id}||{agent}||{question_text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]

def scene_from_video(video_name: str) -> str:
    return Path(video_name).stem if video_name else ""

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

def load_progress(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("next_idx", 0))
    except Exception:
        return 0

def save_progress(path: Path, next_idx: int) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"next_idx": next_idx}, f, indent=2, ensure_ascii=False)
        f.flush()
    tmp.replace(path)


# =========================
# Sanity checks
# =========================
if not ALL_SCENES_JSON.exists():
    raise FileNotFoundError(f"Missing all_scenes.json at: {ALL_SCENES_JSON}")
if not LORA_PATH.exists():
    raise FileNotFoundError(f"Missing LoRA checkpoint at: {LORA_PATH}")
if not VIDEO_ROOT.exists():
    raise FileNotFoundError(f"VIDEO_ROOT not found: {VIDEO_ROOT}")


# =========================
# Load ALL entries (each list element is ONE question)
# =========================
with ALL_SCENES_JSON.open("r", encoding="utf-8") as f:
    all_entries: List[Dict[str, Any]] = json.load(f)

total = len(all_entries)
if total == 0:
    raise ValueError(f"{ALL_SCENES_JSON} is empty.")


# =========================
# Build GLOBAL meta (valid for multiple scenes)
# - only run-level config and dataset-level stats
# =========================
scene_counts: Dict[str, int] = {}
missing_video = 0
for e in all_entries:
    v = e.get("video", "")
    if not v:
        missing_video += 1
        continue
    sid = scene_from_video(v)
    scene_counts[sid] = scene_counts.get(sid, 0) + 1

meta = {
    "all_scenes_json": str(ALL_SCENES_JSON),
    "video_root": str(VIDEO_ROOT),
    "base_model": BASE_MODEL_ID,
    "processor": PROCESSOR_ID,
    "lora_path": str(LORA_PATH),
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_entries_total": total,
    "num_scenes_total": len(scene_counts),
    "missing_video_entries": missing_video,
    "scene_counts": scene_counts,  # aggregated distribution (safe for multi-scene)
}
with META_JSON.open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)


# =========================
# Load model + processor once (keep identical)
# =========================
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))


# =========================
# Resume support (global)
# =========================
start_idx = load_progress(PROGRESS_JSON)
if start_idx >= total:
    print(f"[OK] Nothing to do. next_idx={start_idx}, total={total}")
    raise SystemExit(0)

print(f"[INFO] total_entries={total} start_idx={start_idx}")
print(f"[INFO] num_scenes={len(scene_counts)}")
print(f"[INFO] meta={META_JSON}")
print(f"[INFO] results={RESULTS_JSONL}")
print(f"[INFO] fails={FAILS_JSONL}")
print(f"[INFO] progress={PROGRESS_JSON}")


# =========================
# Main loop: run ONE question per entry, save ONE line immediately
# (No prompt stored; model behavior unchanged)
# =========================
for idx in range(start_idx, total):
    entry = all_entries[idx]
    video_name = entry.get("video", "")
    scene_id = scene_from_video(video_name)
    video_path = VIDEO_ROOT / video_name if video_name else None

    convs = entry.get("conversations", [])
    human_turn = None
    for t in convs:
        if t.get("from") == "human":
            human_turn = t
            break

    agent_name = ""
    question_text = ""

    try:
        if not video_name:
            raise ValueError("Missing 'video' field in entry")
        if video_path is None or not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if human_turn is None:
            raise ValueError("No human turn found in conversations")

        prompt_text = strip_video_marker(human_turn.get("value", ""))
        question_text = parse_question(prompt_text)
        agent_name = parse_agent(question_text)
        qid = make_qid(scene_id, agent_name, question_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        start = time.perf_counter()
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        latency = time.perf_counter() - start

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        record = {
            "qid": qid,
            "scene": scene_id,
            "agent": agent_name,
            "question": question_text,
            "output": output_text,
            "latency_sec": round(latency, 3),
            "entry_idx": idx,
            "status": "ok",
        }

        append_jsonl(RESULTS_JSONL, record)
        save_progress(PROGRESS_JSON, next_idx=idx + 1)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"[{idx+1}/{total}] OK (latest scene={scene_id} agent={agent_name} qid={qid})")

    except Exception as e:
        fail = {
            "entry_idx": idx,
            "scene": scene_id,
            "agent": agent_name,
            "status": "error",
            "error_type": type(e).__name__,
            "error_msg": str(e),
        }
        append_jsonl(FAILS_JSONL, fail)
        save_progress(PROGRESS_JSON, next_idx=idx + 1)

        print(f"[{idx+1}/{total}] ERROR scene={scene_id} agent={agent_name} {type(e).__name__}: {e}")

print(f"[DONE] Appended results to {RESULTS_JSONL}")
