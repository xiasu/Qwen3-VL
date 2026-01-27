import time
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
# default: Load the model on the available device(s)
base_model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto", attn_implementation="flash_attention_2",
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# 1. Video Query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/mmfs1/gscratch/makelab/xia/datasets/capnav/videos/HM3D00000.mp4",
            },
            {"type": "text", "text": "You are given a navigation feasibility question grounded in a scene graph and an agent profile.\n\nTask: Answer whether the agent can move from the START node to the END node.\nReturn only one of:\n  Answer: (A) True | Path: <shortest feasible path>\n  Answer: (B) False | Path: <shortest path> | Fail: <why it fails>\nIf the answer is False, include a short failure reason.\n\n[Question]\nCan [HUMAN] move from Second floor hallway to Master bedroom?\n\n[Agent Profile]\nAgent name: HUMAN\nBody shape: cylinder\nHeight (m): 1.7\nWidth (m): 0.6\nDepth (m): None\nMax vertical cross height (m): 0.3\nCan go up or down stairs: True\nCan operate elevator: True\nCan open the door: True\nDescription: An able-bodied young adult capable of walking, crouching, and stepping over small obstacles up to 0.3 m high.\n\n[Scene Graph Nodes]\nnode_10 — Second floor hallway\nnode_12 — Lobby\nnode_13 — Dining room\nnode_14 — Living room\nnode_15 — Kitchen\nnode_4 — Master bedroom\nnode_5 — Master Bedroom Bathroom\nnode_6 — Guest Bedroom 1\nnode_7 — Guest Bedroom 2\nnode_8 — Crib room\nnode_9 — Second floor shared bathroom\n"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(base_model.device)

# Inference: Generation of the output with timer
start_time = time.perf_counter()
generated_ids = base_model.generate(**inputs, max_new_tokens=1280)
end_time = time.perf_counter()
video_query_time = end_time - start_time

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Base Model Output:")
print(output_text)
print(f"Base Model Video query took {video_query_time:.3f} seconds.")


lora_path = "/mmfs1/gscratch/makelab/xia/Qwen3-VL/output/capnav-test-1/checkpoint-843"
model = PeftModel.from_pretrained(
    base_model,
    lora_path
)

start_time = time.perf_counter()
generated_ids = model.generate(**inputs, max_new_tokens=1280)
end_time = time.perf_counter()
video_query_time = end_time - start_time

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("LoRA Output:")
print(output_text)
print(f"LoRA Video query took {video_query_time:.3f} seconds.")