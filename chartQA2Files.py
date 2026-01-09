import os
import json
from pathlib import Path
from datasets import load_dataset, Image

def main(out_dir="chartqa_qwen", split="train"):
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset; Image(decode=False) avoids decoding to PIL and gives you file path/bytes where possible. :contentReference[oaicite:1]{index=1}
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
    ds = ds.cast_column("image", Image(decode=False))

    records = []

    for i, ex in enumerate(ds):
        # ChartQA columns: image, query, label (list), human_or_machine :contentReference[oaicite:2]{index=2}
        q = ex["query"]
        ans_list = ex["label"]
        ans = ans_list[0] if isinstance(ans_list, list) and len(ans_list) > 0 else str(ans_list)

        img_info = ex["image"]  # typically {'path': ..., 'bytes': ...}
        out_name = f"{split}_{i:08d}.png"
        out_path = img_dir / out_name

        if isinstance(img_info, dict) and img_info.get("path"):
            # copy from cached file
            src = Path(img_info["path"])
            out_path.write_bytes(src.read_bytes())
        elif isinstance(img_info, dict) and img_info.get("bytes") is not None:
            out_path.write_bytes(img_info["bytes"])
        else:
            raise RuntimeError(f"Unexpected image object type: {type(img_info)}")

        records.append({
            "image": str(Path("images") / out_name),
            "conversations": [
                {"from": "human", "value": f"{q}\n<image>"},
                {"from": "gpt", "value": ans}
            ]
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / f"{split}.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} samples")
    print(f"Images: {img_dir}")
    print(f"Annotations: {ann_path}")

if __name__ == "__main__":
    main()