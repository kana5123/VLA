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
NUM_EPISODES = 52
BRIDGE_DATASET_NAME = "bridge_dataset"
BRIDGE_UNNORM_KEY = "bridge_orig"
TFDS_DATA_DIR = Path("/ceph_data/kana5123/bridge_mini")

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path("/ceph_data/kana5123/bridge_v2_data")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_CACHE_DIR = Path("/ceph_data/kana5123/bridge_data_cache")
ATTENTION_RESULTS_DIR = OUTPUT_DIR / "attention_results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
PATCHES_DIR = OUTPUT_DIR / "patches"
METADATA_PATH = DATA_DIR / "metadata.json"

# ── Prompt template (OpenVLA format) ──────────────────────────────────────
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# ── Attention Enhancement Experiment ─────────────────────────────────────
ENHANCEMENT_RESULTS_DIR = OUTPUT_DIR / "enhancement_results"

# Object grounding
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_BOX_THRESHOLD = 0.30
GROUNDING_TEXT_THRESHOLD = 0.25

# Enhancement method hyperparameters
LOGIT_BIAS_ALPHA = 3.0
WEIGHT_RESCALE_LAMBDA = 2.0
HEAD_STEER_TOP_K_HEADS = 8
HEAD_STEER_AMPLIFY = 2.0

# Layer filter: None = all layers, or list of layer indices
ENHANCEMENT_LAYERS = None

# ── V2 Enhancement Experiment (conservative params) ────────────────────
V2_RESULTS_DIR = OUTPUT_DIR / "v2_enhancement_results"

# Conservative hyperparameters (Atlas-scale)
V2_WEIGHT_RESCALE_LAMBDA = 1.1
V2_LOGIT_BIAS_ALPHA = 0.5
V2_HEAD_STEER_TOP_K_HEADS = 4
V2_HEAD_STEER_AMPLIFY = 1.2

# Middle layers only
V2_ENHANCEMENT_LAYERS = [12, 14, 16, 18, 20]

# Background suppression
BG_SUPPRESS_GAMMA = 0.85

# Residual stream steering
RESIDUAL_STEER_ALPHA = 1.5
RESIDUAL_STEER_LAYER = 16

# ── V3 Enhancement Experiment (research-based methods) ──────────────
V3_RESULTS_DIR = OUTPUT_DIR / "v3_enhancement_results"

# ── V4 Corrected Experiment (bug-fixed, filtered) ───────────────────
V4_RESULTS_DIR = OUTPUT_DIR / "v4_corrected_results"

# ── V5 Head Profiling + Targeted Intervention ───────────────────────
HEAD_PROFILING_DIR = OUTPUT_DIR / "head_profiling"
V5_RESULTS_DIR = OUTPUT_DIR / "v5_targeted_results"

# VAR (Visual Attention Redistribution) — ICLR 2025
VAR_SINK_INDICES = [0]          # Vision token 0 is the confirmed attention sink
VAR_P = 0.6                    # Fraction of sink attention to redistribute
VAR_RHO = 0.3                  # Image-centric head selection threshold (filters text-dominant heads)
VAR_BASELINE_RHO = 0.5        # Rho for fixed-var baseline (matches VAR paper's conservative default)
VAR_OBJECT_REDIRECT_WEIGHT = 3.0  # Extra weight for object patches in redistribution
VAR_TEXT_SINK_ENABLED = True        # Also redistribute from text sink ("\n")
VAR_TEXT_SINK_P = 0.3               # Fraction of text sink attention to redistribute
VAR_TEXT_SINK_THRESHOLD = 0.15      # Only redistribute if text max > 15% of total

# Dynamic sink detection (ACT-style α/N threshold)
DYNAMIC_SINK_DETECTION = True      # True = detect per-forward-pass, False = use VAR_SINK_INDICES
SINK_ALPHA = 5.0                   # Threshold multiplier: sink if attn > α/N

# VTR (Vision-Text Rebalance)
VT_SHIFT_FRACTION = 0.3            # Fraction of text attention to shift to vision

# Enhancement layer ranges for layer-selective VAR
LAYER_SELECTIVE_RANGE = (8, 25)    # Apply only in layers 8-24

# Temporal motion-aware attention
TEMPORAL_BOOST_WEIGHT = 2.5        # Weight for motion-detected patches in redistribution
TEMPORAL_DIFF_THRESHOLD = 30.0     # Pixel diff threshold (0-255 range) for motion detection
TEMPORAL_PATCH_MIN_MOTION = 0.15   # Min fraction of patch pixels with motion to select it

# ACT (Attention Calibration Technique) — arXiv 2406.15765
ACT_SINK_ALPHA = 5.0           # Sink threshold: score > alpha / N
ACT_SCALE_BETA = 0.4           # Scale factor for sink tokens

# SPIN (Head Suppression) — EMNLP 2025
SPIN_TOP_K_HEADS = 8           # Number of vision-attending heads to keep
SPIN_SUPPRESS_ALPHA = 0.05     # Suppression factor for non-top-K heads

# ── SAM2 Grounded Segmentation ──────────────────────────────────────
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
GROUNDING_MAX_AREA_FRACTION = 0.5   # Filter bboxes > 50% of image area
GROUNDING_NMS_IOU_THRESHOLD = 0.5   # NMS IoU threshold for dedup
SAM_PATCH_OVERLAP_THRESHOLD = 0.1   # Min mask overlap per grid cell to select patch

# ── Differentiable Attention Adapter ──────────────────────────────
ADAPTER_RESULTS_DIR = OUTPUT_DIR / "adapter_results"
ADAPTER_CHECKPOINT_DIR = ADAPTER_RESULTS_DIR / "checkpoints"
ADAPTER_LOG_DIR = ADAPTER_RESULTS_DIR / "logs"

# Data
ADAPTER_TFRECORD_DIR = Path("/ceph_data/kana5123/bridge_data_v2/0.0.1")
ADAPTER_NUM_TRAIN_EPISODES = None    # None = use all episodes in dataset (auto-detect)
ADAPTER_BATCH_SIZE = 64              # Steps per batch (1 step = 1 image)

# Architecture
ADAPTER_SOURCE_LAYER = 27            # Layer to extract hidden state from
ADAPTER_TARGET_LAYERS = [28, 29, 30, 31]  # Layers where VAR is applied
ADAPTER_NUM_TARGET_LAYERS = 4
ADAPTER_INTERMEDIATE_DIM = 256
ADAPTER_DROPOUT = 0.1

# Training
ADAPTER_LR = 1e-3                    # Increased from 3e-4 for stronger adapter gradient signal
ADAPTER_MIN_LR = 1e-4
ADAPTER_WEIGHT_DECAY = 0.01
ADAPTER_WARMUP_STEPS = 500
ADAPTER_MAX_STEPS = 50000
ADAPTER_GRAD_CLIP = 1.0
ADAPTER_L1_LAMBDA = 0.0001           # Reduced from 0.001: less aggressive sparsity → allows CE gradient to drive learning

# Evaluation
ADAPTER_EVAL_EVERY = 500             # Evaluate every N training steps
ADAPTER_SAVE_EVERY = 2000            # Save checkpoint every N steps
ADAPTER_PATIENCE = 15                # Early stopping patience (in eval intervals)

# ── Adapter v2: Object-Aware Differentiable Adapter ─────────────────
# SAM preprocessing
SAM_MASKS_FILENAME = "object_masks.dat"
SAM_FAILURE_MARKER = 255          # uint8 marker for SAM-failed steps
SAM_EPISODE_FAILURE_THRESHOLD = 0.5  # exclude episode if >50% steps fail

# Cross-attention (Branch 2: redistribution weights)
ADAPTER_V2_QUERY_DIM = 128        # projection dim for query/key
ADAPTER_V2_TEMPERATURE = 2.0      # softmax temperature for smoother gradients

# Blend alpha (proportional → learned transition)
# -1.0 gives sigmoid(-1) ≈ 0.27: stronger initial cross-attention contribution.
# Previous -2.0 (≈0.12) was too conservative, causing slow gradient flow to Branch 2.
ADAPTER_V2_BLEND_INIT = -1.0      # sigmoid(-1) ≈ 0.27

# Object mask embedding (Branch 1 augmentation)
ADAPTER_V2_MASK_DIM = 64          # mask embedding dimension in p-head MLP

# ── LoRA Fine-Tuning Infrastructure ──────────────────────────────────
LORA_R = 16                          # LoRA rank
LORA_ALPHA = 32                      # LoRA alpha (scaling = alpha / r)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # LLaMA attention projections
LORA_DROPOUT = 0.05                  # Dropout on LoRA A/B matrices
LORA_LR = 1e-4                      # LoRA learning rate
LORA_MAX_STEPS = 50000               # Max LoRA fine-tuning steps
LORA_WARMUP_STEPS = 200              # Warmup steps for LoRA
LORA_RESULTS_DIR = OUTPUT_DIR / "lora_results"

# ── LIBERO Simulation Benchmark ──────────────────────────────────────
LIBERO_RESULTS_DIR = OUTPUT_DIR / "libero_results"
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
LIBERO_TASKS_PER_SUITE = 10
LIBERO_EPISODES_PER_TASK = 20      # Rollouts per task for success rate
LIBERO_MAX_STEPS = 300             # Max env steps per episode
LIBERO_IMAGE_SIZE = 256            # Observation image size
LIBERO_ACTION_REPEAT = 1           # Action repeat (1 = no repeat)