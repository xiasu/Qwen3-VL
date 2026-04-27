import time
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# 1. Image Query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
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
inputs = inputs.to(model.device)

# Inference: Generation of the output with timer
start_time = time.perf_counter()
generated_ids = model.generate(**inputs, max_new_tokens=128)
end_time = time.perf_counter()
image_query_time = end_time - start_time

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print(f"Image query took {image_query_time:.3f} seconds.")

print("attn_implementation:", getattr(model.config, "_attn_implementation", None))
print("attn_implementation:", getattr(model.config, "attn_implementation", None))
# import transformers, torch
# from transformers.utils import is_flash_attn_2_available
# print("transformers:", transformers.__version__)
# print("torch:", torch.__version__)
# print("cuda:", torch.version.cuda)
# print("flash_attn_2 available:", is_flash_attn_2_available())

# 2. Video Query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
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
inputs = inputs.to(model.device)

# Inference: Generation of the output with timer
start_time = time.perf_counter()
generated_ids = model.generate(**inputs, max_new_tokens=128)
end_time = time.perf_counter()
video_query_time = end_time - start_time

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print(f"Video query took {video_query_time:.3f} seconds.")