"""Direct reproduction of official OpenVLA LIBERO eval pipeline.
Follows exact code from openvla/experiments/robot/libero/run_libero_eval.py
"""
import os; os.environ["MUJOCO_GL"] = "egl"
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

DEVICE = "cuda:2"

# Load model exactly as official code does
print("Loading model...", flush=True)
from transformers import AutoModelForVision2Seq, AutoProcessor
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    trust_remote_code=True,
)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(DEVICE).eval()
print("Model loaded", flush=True)

# Check norm_stats
print(f"norm_stats keys: {list(model.norm_stats.keys())}", flush=True)

# Setup LIBERO
suite = benchmark.get_benchmark("libero_spatial")()
task = suite.get_task(0)
task_desc = task.language
init_states = suite.get_task_init_states(0)
bddl_file = suite.get_task_bddl_file_path(0)
print(f"Task: {task_desc}", flush=True)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    camera_heights=256, camera_widths=256,
    render_gpu_device_id=3,
)
env.seed(0)

for episode_idx in range(3):
    print(f"\n=== Episode {episode_idx} ===", flush=True)
    env.reset()
    obs = env.set_init_state(init_states[episode_idx])

    # Wait 10 steps
    for _ in range(10):
        obs, _, _, _ = env.step([0,0,0,0,0,0,-1])

    for step in range(300):
        # Get image (official way)
        img = obs["agentview_image"]
        img = img[::-1, ::-1]  # 180 degree rotation
        image = Image.fromarray(img)

        # Get action (official way)
        prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        action = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

        # Post-process gripper
        action[-1] = 2.0 * action[-1] - 1.0  # [0,1] -> [-1,1]
        action[-1] = np.sign(action[-1])  # binarize
        action[-1] = -action[-1]  # invert

        if step < 5 or step % 50 == 0:
            print(f"  Step {step}: action={action}", flush=True)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            print(f"  SUCCESS at step {step}!", flush=True)
            break
    else:
        print(f"  FAILED (300 steps)", flush=True)

env.close()
print("Done!", flush=True)
