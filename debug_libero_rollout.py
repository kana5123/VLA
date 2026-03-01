"""Debug: minimal LIBERO rollout test with timing per step."""
import os; os.environ["MUJOCO_GL"] = "egl"
import sys
import time
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from extract_attention import load_model_from_registry, detokenize_actions

device = "cuda:0"
print("Loading OpenVLA + LoRA...", flush=True)
processor, model, model_cfg = load_model_from_registry("openvla-7b", device)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "outputs/libero_ft/openvla-7b/libero_spatial/lora_adapter")
model.eval()
print("Model loaded", flush=True)

# Create env
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark("libero_spatial")()
task = suite.get_task(0)
bddl_path = suite.get_task_bddl_file_path(0)
init_states = suite.get_task_init_states(0)

print("Creating env...", flush=True)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=256, camera_widths=256,
)
env.seed(0)
env.set_init_state(init_states[0])
obs = env.reset()
print("Env created", flush=True)

instruction = task.language
print(f"Task: {instruction}", flush=True)

# Run 10 steps with timing
for step in range(10):
    t0 = time.time()

    # Get image
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    # Predict action
    t1 = time.time()
    prompt = model_cfg.prompt_template.format(instruction=instruction)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if pv.dtype != model.dtype:
            inputs["pixel_values"] = pv.to(model.dtype)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated = []
    for _ in range(model_cfg.action_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False,
            )
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype),
            ], dim=-1)

    t2 = time.time()

    result = detokenize_actions(model, generated)
    action = result.get("normalized_action")
    action = np.array(action) if action is not None else np.zeros(7)

    t3 = time.time()

    # Step env
    obs, reward, done, info = env.step(action)
    t4 = time.time()

    print(f"  Step {step}: model={t2-t1:.2f}s, detok={t3-t2:.4f}s, env={t4-t3:.3f}s, "
          f"total={t4-t0:.2f}s, action={action[:3].tolist()}", flush=True)

    success_dict = env.check_success()
    task_success = all(success_dict.values()) if isinstance(success_dict, dict) else bool(success_dict)
    if task_success or done:
        print(f"  => Done at step {step}: success={task_success}", flush=True)
        break

env.close()
print("Done!", flush=True)
