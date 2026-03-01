"""Isolate exactly when env.step becomes slow."""
import os; os.environ["MUJOCO_GL"] = "egl"
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, numpy as np, torch
from PIL import Image
from extract_attention import load_model_from_registry

device = "cuda:0"
processor, model, model_cfg = load_model_from_registry("openvla-7b", device)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "outputs/libero_ft/openvla-7b/libero_spatial/lora_adapter")
model.eval()
print("Model loaded on GPU", flush=True)

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark("libero_spatial")()
bddl_path = suite.get_task_bddl_file_path(0)
init_states = suite.get_task_init_states(0)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=256, camera_widths=256,
)
env.seed(0); env.set_init_state(init_states[0]); obs = env.reset()

# Phase 1: env.step with model loaded but no forward yet
print("\nPhase 1: env.step with model loaded (no forward):", flush=True)
for i in range(3):
    action = np.random.uniform(-0.1, 0.1, 7)
    t0 = time.time()
    obs, r, d, info = env.step(action)
    print(f"  Step {i}: {time.time()-t0:.3f}s", flush=True)

# Phase 2: do ONE model forward
print("\nPhase 2: Running one model forward...", flush=True)
img = obs["agentview_image"]
if img.dtype != np.uint8:
    img = (img * 255).astype(np.uint8)
image = Image.fromarray(img)
prompt = model_cfg.prompt_template.format(instruction="test")
inputs = processor(prompt, image, return_tensors="pt").to(device)
pv = inputs.get("pixel_values")
if pv is not None and pv.dtype != model.dtype:
    inputs["pixel_values"] = pv.to(model.dtype)
with torch.no_grad():
    out = model(**inputs, use_cache=False)
torch.cuda.synchronize()
print("Forward done", flush=True)

print("\nPhase 3: env.step AFTER model forward:", flush=True)
for i in range(3):
    action = np.random.uniform(-0.1, 0.1, 7)
    t0 = time.time()
    obs, r, d, info = env.step(action)
    print(f"  Step {i}: {time.time()-t0:.3f}s", flush=True)

# Phase 4: delete output tensor, clear cache
del out
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("\nPhase 4: env.step AFTER clearing cache:", flush=True)
for i in range(3):
    action = np.random.uniform(-0.1, 0.1, 7)
    t0 = time.time()
    obs, r, d, info = env.step(action)
    print(f"  Step {i}: {time.time()-t0:.3f}s", flush=True)

env.close()
print("Done!", flush=True)
