"""Debug: isolate MuJoCo env.step() speed issue."""
import os; os.environ["MUJOCO_GL"] = "egl"
import time
import numpy as np

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

action = np.zeros(7)

# Test 1: env.step with random actions
print("\nTest 1: env.step() timing (20 steps):")
times = []
for i in range(20):
    action = np.random.uniform(-0.1, 0.1, 7)
    t0 = time.time()
    obs, r, d, info = env.step(action)
    t1 = time.time()
    times.append(t1 - t0)
    print(f"  Step {i}: {t1-t0:.3f}s", flush=True)

print(f"\nMean: {np.mean(times):.3f}s, Median: {np.median(times):.3f}s")
print(f"Min: {np.min(times):.3f}s, Max: {np.max(times):.3f}s")

env.close()

# Test 2: smaller image size
print("\nTest 2: 128x128 image size:")
env2 = OffScreenRenderEnv(
    bddl_file_name=bddl_path, has_renderer=False, has_offscreen_renderer=True,
    render_camera="agentview", use_camera_obs=True, camera_heights=128, camera_widths=128,
)
env2.seed(0)
env2.set_init_state(init_states[0])
obs2 = env2.reset()

times2 = []
for i in range(20):
    action = np.random.uniform(-0.1, 0.1, 7)
    t0 = time.time()
    obs2, r, d, info = env2.step(action)
    t1 = time.time()
    times2.append(t1 - t0)
    if i < 3:
        print(f"  Step {i}: {t1-t0:.3f}s", flush=True)
print(f"Mean: {np.mean(times2):.3f}s, Image shape: {obs2['agentview_image'].shape}")
env2.close()

print("\nDone!")
