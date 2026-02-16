"""Configuration for OpenVLA Attention Analysis Pipeline."""

from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_NAME = "openvla/openvla-7b"
TORCH_DTYPE = "bfloat16"

# ── Model architecture ────────────────────────────────────────────────────
NUM_LAYERS = 32
NUM_HEADS = 32
VISION_GRID_SIZE = 16  # 16x16 = 256 vision tokens (per encoder; Prismatic uses dual encoder)

# ── Analysis parameters ───────────────────────────────────────────────────
TOP_K = 5
NUM_ACTION_TOKENS = 7
ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# ── Data ──────────────────────────────────────────────────────────────────
NUM_EPISODES = 10
BRIDGE_DATASET_NAME = "bridge_dataset"
BRIDGE_UNNORM_KEY = "bridge_orig"

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "bridge_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ATTENTION_RESULTS_DIR = OUTPUT_DIR / "attention_results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
PATCHES_DIR = OUTPUT_DIR / "patches"
METADATA_PATH = DATA_DIR / "metadata.json"

# ── Prompt template (OpenVLA format) ──────────────────────────────────────
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
