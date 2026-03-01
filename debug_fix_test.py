"""Test: predict_action with del+empty_cache → env.step speed."""
import os; os.environ["MUJOCO_GL"] = "egl"
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, numpy as np, torch
from PIL import Image
from extract_attention import load_model_from_registry, detokenize_actions
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

device = "cuda:0"
processor, model, model_cfg = load_model_from_registry("openvla-7b", device)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "outputs/libero_ft/openvla-7b/libero_spatial/lora_adapter")
model.eval()
print("Model loaded", flush=True)

suite = benchmark.get_benchmark("libero_spatial")()
bddl_path = suite.get_task_bddl_file_path(0)
init_states = suite.get_task_init_states(0)
instruction = suite.get_task(0).language
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=256, camera_widths=256,
)
env.seed(0); env.set_init_state(init_states[0]); obs = env.reset()

for step in range(10):
    t0 = time.time()
    img = obs["agentview_image"]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    # predict_action with cleanup
    t_model = time.time()
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       pixel_values=pixel_values, use_cache=False)
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask,
                torch.ones(1, 1, device=device, dtype=attention_mask.dtype)], dim=-1)

    result = detokenize_actions(model, generated)
    action = np.array(result.get("normalized_action", [0]*7))

    # Cleanup GPU
    del outputs, inputs, input_ids, attention_mask, pixel_values, next_tok
    torch.cuda.empty_cache()
    t_model_end = time.time()

    # env step
    t_env = time.time()
    obs, r, d, info = env.step(action)
    t_env_end = time.time()

    print(f"  Step {step}: model={t_model_end-t_model:.2f}s, env={t_env_end-t_env:.3f}s, "
          f"total={t_env_end-t0:.2f}s", flush=True)

env.close()
print("Done!", flush=True)
