"""Quick check: official FT model action predictions on LIBERO."""
import os; os.environ["MUJOCO_GL"] = "egl"
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch
from PIL import Image
from extract_attention import load_model_from_registry, detokenize_actions
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

device = "cuda:0"
processor, model, model_cfg = load_model_from_registry("openvla-7b-ft-libero", device)
model.eval()
print(f"Model loaded on {device}", flush=True)

# Check norm_stats
norm_stats = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
if norm_stats:
    print(f"  norm_stats keys: {list(norm_stats.keys())}", flush=True)
else:
    print("  WARNING: no norm_stats found!", flush=True)

suite = benchmark.get_benchmark("libero_spatial")()
bddl_path = suite.get_task_bddl_file_path(0)
init_states = suite.get_task_init_states(0)
instruction = suite.get_task(0).language
print(f"Instruction: {instruction}", flush=True)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=256, camera_widths=256,
    render_gpu_device_id=1,
)
env.seed(0); env.set_init_state(init_states[0]); obs = env.reset()

prompt = model_cfg.prompt_template.format(instruction=instruction)
print(f"Prompt: {prompt}", flush=True)

for step in range(10):
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != model.dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")

    generated = []
    for _ in range(model_cfg.action_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       pixel_values=pixel_values, use_cache=False)
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype)], dim=-1)

    # Decode with libero_spatial stats
    result = detokenize_actions(model, generated, unnorm_key="libero_spatial")
    norm_action = np.array(result["normalized_action"])
    unnorm_action = result.get("unnormalized_action")
    print(f"Step {step}: tokens={generated}, bins={result['bin_indices']}", flush=True)
    print(f"  norm_action={norm_action}", flush=True)
    if unnorm_action is not None:
        unnorm_action = np.array(unnorm_action)
        print(f"  unnorm_action={unnorm_action}", flush=True)
    else:
        print(f"  unnorm_action=None (using norm as fallback)", flush=True)
        unnorm_action = norm_action

    del outputs, inputs, input_ids, attention_mask, pixel_values, next_tok
    torch.cuda.empty_cache()

    obs, r, d, info = env.step(unnorm_action)
    success_dict = env.check_success()
    task_success = all(success_dict.values()) if isinstance(success_dict, dict) else bool(success_dict)
    print(f"  reward={r}, done={d}, success={task_success}", flush=True)
    if task_success:
        print("SUCCESS! Task completed.", flush=True)
        break

env.close()
print("Done!", flush=True)
