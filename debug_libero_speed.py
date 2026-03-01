"""Quick debug test: measure predict_action speed with KV cache."""
import os; os.environ["MUJOCO_GL"] = "egl"
import time, torch, numpy as np
from PIL import Image
from extract_attention import load_model_from_registry, detokenize_actions

device = "cuda:0"
print("Loading OpenVLA + LoRA...")
processor, model, model_cfg = load_model_from_registry("openvla-7b", device)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "outputs/libero_ft/openvla-7b/libero_spatial/lora_adapter")
model.eval()
print("Model loaded")

img = Image.new("RGB", (256, 256), (128, 128, 128))
instruction = "pick up the black bowl"
prompt = model_cfg.prompt_template.format(instruction=instruction)
inputs = processor(prompt, img, return_tensors="pt").to(device)
if "pixel_values" in inputs:
    pv = inputs["pixel_values"]
    if pv.dtype != model.dtype:
        inputs["pixel_values"] = pv.to(model.dtype)

input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask")
pixel_values = inputs.get("pixel_values")

# Method 1: use_cache=True with KV cache reuse
generated = []
past_key_values = None
t_start = time.time()
am = attention_mask.clone() if attention_mask is not None else None
for i in range(7):
    with torch.no_grad():
        if i == 0:
            outputs = model(input_ids=input_ids, attention_mask=am,
                          pixel_values=pixel_values, use_cache=True)
        else:
            outputs = model(input_ids=next_tok, attention_mask=am,
                          past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_tok.item())
    if am is not None:
        am = torch.cat([am, torch.ones(1, 1, device=device, dtype=am.dtype)], dim=-1)

t_end = time.time()
print(f"KV cache method: 7 tokens in {t_end-t_start:.2f}s: {generated}")
result = detokenize_actions(model, generated)
print(f"  normalized_action: {result.get('normalized_action')}")
print(f"  unnormalized_action: {result.get('unnormalized_action')}")

# Method 2: use_cache=False (old method)
generated2 = []
t_start2 = time.time()
ids2 = input_ids.clone()
am2 = attention_mask.clone() if attention_mask is not None else None
for i in range(7):
    with torch.no_grad():
        outputs2 = model(input_ids=ids2, attention_mask=am2,
                        pixel_values=pixel_values, use_cache=False)
    next_tok2 = outputs2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated2.append(next_tok2.item())
    ids2 = torch.cat([ids2, next_tok2], dim=-1)
    if am2 is not None:
        am2 = torch.cat([am2, torch.ones(1, 1, device=device, dtype=am2.dtype)], dim=-1)

t_end2 = time.time()
print(f"\nNo-cache method: 7 tokens in {t_end2-t_start2:.2f}s: {generated2}")

# Method 3: model.generate()
t_start3 = time.time()
with torch.no_grad():
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=7,
        do_sample=False,
    )
new_tokens = gen_ids[0, input_ids.shape[1]:].tolist()
t_end3 = time.time()
print(f"\nmodel.generate(): 7 tokens in {t_end3-t_start3:.2f}s: {new_tokens}")
result3 = detokenize_actions(model, new_tokens[:7])
print(f"  normalized_action: {result3.get('normalized_action')}")

# Quick MuJoCo env test
print("\nTesting MuJoCo env step speed...")
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark("libero_spatial")()
task = suite.get_task(0)
bddl_path = suite.get_task_bddl_file_path(0)
init_states = suite.get_task_init_states(0)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=256, camera_widths=256,
)
env.seed(0)
env.set_init_state(init_states[0])
obs = env.reset()

action = np.array(result.get("normalized_action", [0]*7))
t4 = time.time()
for _ in range(10):
    obs, r, d, info = env.step(action)
t5 = time.time()
print(f"10 env steps: {t5-t4:.2f}s ({(t5-t4)/10*1000:.0f}ms/step)")
env.close()

print(f"\nEstimate: 300 steps/episode = {300*(t_end-t_start)/7 + 300*(t5-t4)/10:.0f}s per episode (KV cache)")
print(f"Estimate: 300 steps/episode = {300*(t_end2-t_start2)/7 + 300*(t5-t4)/10:.0f}s per episode (no cache)")
