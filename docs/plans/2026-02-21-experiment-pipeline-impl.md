# Experiment Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end experiment pipeline that trains 4 adapter configurations (base, v1, v2-prop, v2-full), evaluates offline with MSE metrics, compares results, and prepares LoRA infrastructure.

**Architecture:** Incremental modification of existing working code. Three files modified (`adapter_eval.py`, `adapter_train.py`, `config.py`), three files created (`run_adapter_experiment.py`, `compare_adapter_results.py`, `adapter_lora.py`). All evaluation is offline (no robot inference). The experiment runner orchestrates training + eval via subprocess calls.

**Tech Stack:** PyTorch, Accelerate (DDP), matplotlib, peft (LoRA), numpy. Existing: `attention_v3.py` (VAR engine), `adapter_model.py` (V1+V2 adapters), `adapter_data.py` (data pipeline with SAM masks).

---

## Task 1: adapter_eval.py — Auto-detect adapter version from checkpoint

**Files:**
- Modify: `adapter_eval.py:23,54-58`

**Context:** Currently `adapter_eval.py` hardcodes `AttentionAdapter` (v1). Checkpoints from `adapter_train.py` save `ckpt["config"]["adapter_version"]` (line 370 of adapter_train.py). We need to auto-detect and instantiate the correct class.

**Step 1: Add v2 import**

In `adapter_eval.py:23`, change:
```python
from adapter_model import AttentionAdapter
```
to:
```python
from adapter_model import AttentionAdapter, AttentionAdapterV2
```

**Step 2: Replace hardcoded v1 adapter creation**

In `adapter_eval.py:54-58`, replace:
```python
ckpt = torch.load(checkpoint_path, map_location=device)
hidden_dim = self.model.config.text_config.hidden_size
self.adapter = AttentionAdapter(hidden_dim=hidden_dim).to(device)
self.adapter.load_state_dict(ckpt["adapter_state_dict"])
self.adapter.eval()
print(f"Adapter loaded from {checkpoint_path} (step {ckpt.get('global_step', '?')})")
```
with:
```python
ckpt = torch.load(checkpoint_path, map_location=device)
hidden_dim = self.model.config.text_config.hidden_size
self.adapter_version = ckpt.get("config", {}).get("adapter_version", 1)

if self.adapter_version == 2:
    self.adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        vision_tokens=self.vision_end,
    ).to(device)
else:
    self.adapter = AttentionAdapter(hidden_dim=hidden_dim).to(device)

self.adapter.load_state_dict(ckpt["adapter_state_dict"])
self.adapter.eval()
print(
    f"Adapter v{self.adapter_version} loaded from {checkpoint_path} "
    f"(step {ckpt.get('global_step', '?')})"
)
```

**Step 3: Verify**

Run: `python -c "from adapter_eval import AdapterEvaluator; print('import OK')"`
Expected: `import OK` (no runtime test — no checkpoint available yet)

**Step 4: Commit**

```bash
git add adapter_eval.py
git commit -m "feat(eval): auto-detect adapter v1/v2 from checkpoint"
```

---

## Task 2: adapter_eval.py — Upgrade get_v3_ctx_for_eval() for v2

**Files:**
- Modify: `adapter_eval.py:222-267`

**Context:** The current `get_v3_ctx_for_eval()` only captures `h[:, -1, :]` (h_last) and calls `adapter(captured["h"].float())` — the v1 signature. For v2, it must also capture `h_vision`, accept `object_mask`, compute redistribution weights with `blend_alpha` blending, and set `ctx.redistribution_weights`. The blending logic is already proven in `adapter_train.py:119-145`.

**Step 1: Replace the full function**

Replace `adapter_eval.py:222-267` (the entire `get_v3_ctx_for_eval` function) with:

```python
def get_v3_ctx_for_eval(
    model, adapter, device, vision_end, adapter_enabled, inputs,
    adapter_version=1, object_mask=None,
):
    """Create V3Context with adapter-predicted p values for evaluation.

    For v2 adapters, also computes redistribution_weights via cross-attention
    blended with proportional fallback using blend_alpha.
    """
    if not adapter_enabled:
        return None

    # Get hidden state from layer 27
    captured = {}

    def hook_fn(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h_last"] = h[:, -1, :]
        captured["h_vision"] = h[:, :vision_end, :]

    if hasattr(model, "language_model"):
        hook_layer = model.language_model.model.layers[config.ADAPTER_SOURCE_LAYER]
    else:
        hook_layer = model.model.layers[config.ADAPTER_SOURCE_LAYER]

    hook = hook_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**{k: v for k, v in inputs.items()}, use_cache=False)
    hook.remove()

    # Adapter prediction
    redistribution_weights = None

    with torch.no_grad():
        if adapter_version == 2:
            h_last = captured["h_last"].float()
            h_vision = captured["h_vision"].float()

            mask_tensor = None
            if object_mask is not None:
                mask_tensor = torch.from_numpy(object_mask).float().unsqueeze(0).to(device)

            p_matrix, redist_raw = adapter(h_last, h_vision, mask_tensor)

            # Blend learned redistribution with proportional (same as adapter_train.py)
            if redist_raw is not None:
                blend = adapter.blend_alpha
                V = vision_end
                prop_weights = torch.ones(1, V, device=device, dtype=torch.float32)
                for si in config.VAR_SINK_INDICES:
                    if si < V:
                        prop_weights[0, si] = 0.0
                prop_weights = prop_weights / prop_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                final_redist = blend * redist_raw + (1 - blend) * prop_weights
                redistribution_weights = final_redist.squeeze(0)  # (V,)
        else:
            p_matrix = adapter(captured["h_last"].float())  # (1, 4, 32)

    # Build full p tensor
    full_p = torch.zeros(
        config.NUM_LAYERS, config.NUM_HEADS, device=device, dtype=p_matrix.dtype,
    )
    _target_idx = torch.tensor(
        config.ADAPTER_TARGET_LAYERS, device=device,
    ).unsqueeze(1).expand(-1, config.NUM_HEADS)
    full_p = full_p.scatter(0, _target_idx, p_matrix[0])

    ctx = V3Context(
        active=True,
        use_var=True,
        var_p=config.VAR_P,
        var_rho=config.VAR_RHO,
        var_sink_indices=list(config.VAR_SINK_INDICES),
        vision_end=vision_end,
        enhancement_layers=set(config.ADAPTER_TARGET_LAYERS),
        per_head_var_strength=full_p,
        redistribution_weights=redistribution_weights,
    )
    return ctx
```

**Step 2: Verify syntax**

Run: `python -c "from adapter_eval import get_v3_ctx_for_eval; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add adapter_eval.py
git commit -m "feat(eval): upgrade get_v3_ctx_for_eval for v2 adapter with redistribution weights"
```

---

## Task 3: adapter_eval.py — Add object mask loading to _run_inference()

**Files:**
- Modify: `adapter_eval.py:37` (add instance var), `adapter_eval.py:64-79` (modify _run_inference)

**Context:** For v2 evaluation, `_run_inference` must load the SAM object mask for the current step from the memmap file (`object_masks.dat`). The memmap format is `(N, 256)` uint8, where N = total dataset steps. The mask loading pattern is in `adapter_data.py:104-118`.

**Step 1: Add memmap loading in __init__**

After line 62 (`self.tokenizer = ActionTokenizer(self.model)`), add:

```python
        # ── Object masks memmap (v2 only) ──
        self.masks_mmap = None
        if self.adapter_version == 2:
            masks_path = config.DATA_CACHE_DIR / config.SAM_MASKS_FILENAME
            if masks_path.exists():
                total_steps = np.memmap(
                    str(config.DATA_CACHE_DIR / "images.dat"), dtype=np.uint8, mode="r",
                ).shape[0] // (256 * 256 * 3)
                self.masks_mmap = np.memmap(
                    str(masks_path), dtype=np.uint8, mode="r",
                    shape=(total_steps, config.VISION_GRID_SIZE ** 2),
                )
                print(f"  Object masks loaded: {masks_path} ({total_steps} steps)")
            else:
                print(f"  WARNING: No object masks at {masks_path}, v2 eval without masks")
```

**Step 2: Add global_step_id to evaluate() loop and pass object_mask**

In `adapter_eval.py:136-145`, the evaluate loop iterates over batches. We need to pass `object_mask` to `_run_inference`. Modify `_run_inference` signature and the call site.

Change `_run_inference` signature (line 64) from:
```python
def _run_inference(self, image, instruction, adapter_enabled: bool) -> dict:
```
to:
```python
def _run_inference(self, image, instruction, adapter_enabled: bool, object_mask=None) -> dict:
```

Change the call to `get_v3_ctx_for_eval` (lines 75-79) from:
```python
        ctx = get_v3_ctx_for_eval(
            self.model, self.adapter, self.device, self.vision_end,
            adapter_enabled=adapter_enabled,
            inputs=inputs,
        ) if adapter_enabled else None
```
to:
```python
        ctx = get_v3_ctx_for_eval(
            self.model, self.adapter, self.device, self.vision_end,
            adapter_enabled=adapter_enabled,
            inputs=inputs,
            adapter_version=self.adapter_version,
            object_mask=object_mask,
        ) if adapter_enabled else None
```

In `evaluate()`, change lines 136-145 to pass `global_step_ids` and object masks:

```python
        for batch in tqdm_wrapper(test_loader, desc="Eval"):
            image = batch["images"][0]
            instruction = batch["instructions"][0]
            gt_action = batch["actions"][0]
            ep_id = batch["episode_ids"][0]
            step_id = batch["step_ids"][0]

            # Load object mask for this step if available
            obj_mask = None
            if self.masks_mmap is not None and "global_step_ids" in batch:
                gid = batch["global_step_ids"][0]
                mask = self.masks_mmap[gid]
                if mask.max() != config.SAM_FAILURE_MARKER:
                    obj_mask = mask

            for condition in ["baseline", "adapter"]:
                adapter_on = condition == "adapter"
                pred = self._run_inference(
                    image, instruction, adapter_enabled=adapter_on,
                    object_mask=obj_mask if adapter_on else None,
                )
```

**Step 3: Add --output_dir CLI option**

In `main()` (line 279-291), add output directory support:

Replace:
```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention Adapter")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=100)
    args = parser.parse_args()

    evaluator = AdapterEvaluator(args.checkpoint, device=args.device)
    evaluator.evaluate(num_episodes=args.num_episodes)
```
with:
```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention Adapter")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for eval results")
    args = parser.parse_args()

    evaluator = AdapterEvaluator(args.checkpoint, device=args.device)
    results = evaluator.evaluate(num_episodes=args.num_episodes)

    # Override output path if specified
    if args.output_dir:
        eval_dir = Path(args.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_results = {k: v for k, v in results.items() if k != "per_step"}
        for cond in ["baseline", "adapter"]:
            if cond in summary_results:
                summary_results[cond] = {
                    k: v for k, v in summary_results[cond].items() if k != "per_step"
                }
        out_path = eval_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"Results also saved: {out_path}")
```

**Step 4: Verify syntax**

Run: `python -c "from adapter_eval import AdapterEvaluator; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add adapter_eval.py
git commit -m "feat(eval): add object mask loading, output_dir override, global_step_id support"
```

---

## Task 4: adapter_train.py — Add --freeze_blend, --adapter_version, --output_dir flags

**Files:**
- Modify: `adapter_train.py:407-412` (argparse), `adapter_train.py:455-460` (adapter creation), `adapter_train.py:355` (checkpoint path), `adapter_train.py:529-532` (output dirs)

**Context:** The experiment runner needs to control: (1) which adapter version to create, (2) whether to freeze blend_alpha at 0, (3) where to save outputs. Currently `adapter_train.py` always creates V2, saves to `config.ADAPTER_CHECKPOINT_DIR`, and has no blend freezing.

**Step 1: Add CLI arguments**

After line 411 (`parser.add_argument("--resume"...)`), add:

```python
    parser.add_argument("--adapter_version", type=int, default=2,
                        choices=[1, 2], help="Adapter version (1=MLP only, 2=object-aware)")
    parser.add_argument("--freeze_blend", action="store_true",
                        help="Freeze blend_alpha at 0 (v2-prop: proportional redistribution only)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (checkpoints + logs saved here)")
```

**Step 2: Use adapter_version to select model**

Replace lines 455-462 (adapter creation section):
```python
    # ── Adapter model ──
    hidden_dim = model.config.text_config.hidden_size  # 4096
    adapter = AttentionAdapterV2(
        hidden_dim=hidden_dim,
        vision_tokens=vision_end,
    )
    if is_main:
        print(f"Adapter parameters: {adapter.param_count():,}")
```
with:
```python
    # ── Adapter model ──
    hidden_dim = model.config.text_config.hidden_size  # 4096
    if args.adapter_version == 2:
        adapter = AttentionAdapterV2(
            hidden_dim=hidden_dim,
            vision_tokens=vision_end,
        )
    else:
        adapter = AttentionAdapter(hidden_dim=hidden_dim)

    # Freeze blend_alpha if requested (v2-prop config)
    if args.freeze_blend and args.adapter_version == 2:
        raw = adapter
        raw._blend_logit.requires_grad_(False)
        # Force blend_alpha to 0 by setting logit to -20 (sigmoid(-20) ~ 2e-9)
        with torch.no_grad():
            raw._blend_logit.fill_(-20.0)
        if is_main:
            print(f"  blend_alpha FROZEN at {raw.blend_alpha.item():.6f}")

    if is_main:
        print(f"Adapter v{args.adapter_version} parameters: {adapter.param_count():,}")
```

**Step 3: Use output_dir for checkpoint/log paths**

After the adapter creation section, before `# ── Optimizer + Scheduler ──`, add override logic. And modify the output dirs section (lines 529-532).

Replace lines 529-532:
```python
    # ── Create output dirs ──
    if is_main:
        config.ADAPTER_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.ADAPTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
```
with:
```python
    # ── Create output dirs ──
    if args.output_dir:
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        log_dir = Path(args.output_dir) / "logs"
    else:
        ckpt_dir = config.ADAPTER_CHECKPOINT_DIR
        log_dir = config.ADAPTER_LOG_DIR
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
```

**Step 4: Update save_checkpoint calls to use ckpt_dir**

The `save_checkpoint` function (line 341) uses `config.ADAPTER_CHECKPOINT_DIR`. We need to pass the directory. Change `save_checkpoint` signature to accept `checkpoint_dir` parameter:

Replace `save_checkpoint` function (lines 341-377):
```python
def save_checkpoint(
    adapter,
    optimizer,
    scheduler,
    global_step: int,
    best_val_loss: float,
    patience_counter: int,
    accelerator: Accelerator,
    filename: str,
    checkpoint_dir: Path = None,
):
    """Save adapter checkpoint (main process only)."""
    if not accelerator.is_main_process:
        return

    save_dir = checkpoint_dir or config.ADAPTER_CHECKPOINT_DIR
    path = save_dir / filename
    raw_adapter = accelerator.unwrap_model(adapter)
    is_v2 = isinstance(raw_adapter, AttentionAdapterV2)
    torch.save({
        "adapter_state_dict": raw_adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": {
            "lr": config.ADAPTER_LR,
            "num_target_layers": config.ADAPTER_NUM_TARGET_LAYERS,
            "target_layers": config.ADAPTER_TARGET_LAYERS,
            "source_layer": config.ADAPTER_SOURCE_LAYER,
            "l1_lambda": config.ADAPTER_L1_LAMBDA,
            "adapter_version": 2 if is_v2 else 1,
            **({"query_dim": config.ADAPTER_V2_QUERY_DIM,
                "temperature": config.ADAPTER_V2_TEMPERATURE,
                "blend_init": config.ADAPTER_V2_BLEND_INIT,
                "mask_dim": config.ADAPTER_V2_MASK_DIM,
               } if is_v2 else {}),
        },
    }, path)
    print(f"  Checkpoint saved: {path}")
```

**Step 5: Update all save_checkpoint call sites**

Find all calls to `save_checkpoint` in the training loop and add `checkpoint_dir=ckpt_dir`:

- Line 642-644: `save_checkpoint(..., "best.pt")` → add `checkpoint_dir=ckpt_dir`
- Line 661-666: `save_checkpoint(..., f"step_{global_step}.pt")` → add `checkpoint_dir=ckpt_dir`
- Line 686-688: `save_checkpoint(..., "final.pt")` → add `checkpoint_dir=ckpt_dir`

Also update the training log save path (line 692):
```python
        log_path = log_dir / "training_log.json"
```

**Step 6: Handle v1 adapter in data loading (skip object masks)**

In lines 491-497 (data loader creation), change to respect adapter_version:
```python
    train_loader, val_loader, _ = create_dataloaders(
        num_episodes=args.num_episodes,
        batch_size=per_gpu_bs,
        source="tfrecord",
        accelerator=accelerator,
        use_object_masks=(args.adapter_version == 2),
    )
```

**Step 7: Verify syntax**

Run: `python -c "from adapter_train import train; print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add adapter_train.py
git commit -m "feat(train): add --adapter_version, --freeze_blend, --output_dir flags"
```

---

## Task 5: run_adapter_experiment.py — Create experiment runner

**Files:**
- Create: `run_adapter_experiment.py`

**Context:** Automates training and evaluation of 4 configurations: base (no adapter, eval only), v1 (AttentionAdapter), v2-prop (V2 with frozen blend), v2-full (V2 with learnable blend). Uses subprocess to launch `adapter_train.py` and `adapter_eval.py` for each config.

**Step 1: Write the experiment runner**

Create `run_adapter_experiment.py`:

```python
"""Run the full adapter experiment: train + eval for 4 configurations.

Configurations:
    base    — No adapter, raw OpenVLA baseline (eval only)
    v1      — AttentionAdapter (MLP-only, no object masks)
    v2-prop — AttentionAdapterV2, blend_alpha frozen at 0 (proportional redistribution)
    v2-full — AttentionAdapterV2, blend_alpha learnable (SAM masks + learned redistribution)

Usage:
    python run_adapter_experiment.py
    python run_adapter_experiment.py --configs v1 v2-full
    python run_adapter_experiment.py --skip_training   # eval only (checkpoints must exist)
    python run_adapter_experiment.py --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config

# ── Experiment Configurations ────────────────────────────────────────────

EXPERIMENT_DIR = config.OUTPUT_DIR / "experiment_results"

CONFIGS = {
    "base": {
        "skip_training": True,
        "adapter_version": None,
        "description": "Raw OpenVLA baseline (no adapter)",
    },
    "v1": {
        "adapter_version": 1,
        "freeze_blend": False,
        "description": "AttentionAdapter v1 (MLP only)",
    },
    "v2-prop": {
        "adapter_version": 2,
        "freeze_blend": True,
        "description": "AttentionAdapterV2, proportional redistribution (blend frozen)",
    },
    "v2-full": {
        "adapter_version": 2,
        "freeze_blend": False,
        "description": "AttentionAdapterV2, learned redistribution (blend learnable)",
    },
}


def run_command(cmd: list[str], description: str, log_path: Path | None = None) -> int:
    """Run a command, streaming output to both stdout and optional log file."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            proc.wait()
            return proc.returncode
    else:
        return subprocess.call(cmd)


def train_config(name: str, cfg: dict, gpus: str, num_episodes: int | None) -> bool:
    """Train a single adapter configuration. Returns True on success."""
    if cfg.get("skip_training"):
        print(f"[{name}] Skipping training (eval-only config)")
        return True

    output_dir = EXPERIMENT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = []
    gpu_list = gpus.split(",")
    n_gpus = len(gpu_list)

    if n_gpus > 1:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(n_gpus),
            "adapter_train.py",
        ]
    else:
        cmd = ["python", "adapter_train.py"]

    cmd.extend([
        "--adapter_version", str(cfg["adapter_version"]),
        "--output_dir", str(output_dir),
    ])

    if cfg.get("freeze_blend"):
        cmd.append("--freeze_blend")

    if num_episodes is not None:
        cmd.extend(["--num_episodes", str(num_episodes)])

    env_prefix = f"CUDA_VISIBLE_DEVICES={gpus}"
    full_cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpus}"] + cmd

    log_path = output_dir / "train.log"
    t0 = time.time()
    rc = run_command(full_cmd, f"Training [{name}]: {cfg['description']}", log_path)
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{name}] Training FAILED (exit code {rc}) after {elapsed:.0f}s")
        return False

    print(f"[{name}] Training completed in {elapsed:.0f}s")
    return True


def eval_config(
    name: str, cfg: dict, device: str, num_eval_episodes: int,
) -> bool:
    """Evaluate a single configuration. Returns True on success."""
    output_dir = EXPERIMENT_DIR / name
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("skip_training"):
        # Base config: run eval without adapter
        # We use adapter_eval.py in a special mode — the base eval doesn't need
        # an adapter checkpoint. We'll handle this by running inference for
        # baseline only (no adapter condition).
        print(f"[{name}] Running baseline-only evaluation...")
        cmd = [
            "python", "adapter_eval.py",
            "--checkpoint", "NONE",  # special marker
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
            "--baseline_only",
        ]
    else:
        # Find best checkpoint
        ckpt_dir = output_dir / "checkpoints"
        best_ckpt = ckpt_dir / "best.pt"
        if not best_ckpt.exists():
            # Fallback to final
            best_ckpt = ckpt_dir / "final.pt"
        if not best_ckpt.exists():
            print(f"[{name}] ERROR: No checkpoint found in {ckpt_dir}")
            return False

        cmd = [
            "python", "adapter_eval.py",
            "--checkpoint", str(best_ckpt),
            "--device", device,
            "--num_episodes", str(num_eval_episodes),
            "--output_dir", str(eval_dir),
        ]

    log_path = output_dir / "eval.log"
    t0 = time.time()
    rc = run_command(cmd, f"Evaluating [{name}]: {cfg['description']}", log_path)
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{name}] Evaluation FAILED (exit code {rc}) after {elapsed:.0f}s")
        return False

    print(f"[{name}] Evaluation completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full adapter experiment")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="GPU IDs for training (comma-separated)")
    parser.add_argument("--eval_device", type=str, default="cuda:0",
                        help="Device for evaluation")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Training episodes (None=all)")
    parser.add_argument("--num_eval_episodes", type=int, default=200,
                        help="Evaluation episodes")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, eval only (checkpoints must exist)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, training only")
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"  ADAPTER EXPERIMENT PIPELINE")
    print(f"  Configs: {args.configs}")
    print(f"  GPUs (train): {args.gpus}")
    print(f"  Eval device: {args.eval_device}")
    print(f"{'#' * 60}\n")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = {}
    t_total = time.time()

    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n{'*' * 60}")
        print(f"  CONFIG: {name} — {cfg['description']}")
        print(f"{'*' * 60}")

        # Training phase
        if not args.skip_training:
            ok = train_config(name, cfg, args.gpus, args.num_episodes)
            if not ok and not cfg.get("skip_training"):
                results_summary[name] = {"status": "train_failed"}
                continue

        # Evaluation phase
        if not args.skip_eval:
            ok = eval_config(name, cfg, args.eval_device, args.num_eval_episodes)
            if ok:
                eval_path = EXPERIMENT_DIR / name / "eval" / "eval_results.json"
                if eval_path.exists():
                    results_summary[name] = json.loads(eval_path.read_text())
                    results_summary[name]["status"] = "complete"
                else:
                    results_summary[name] = {"status": "eval_no_output"}
            else:
                results_summary[name] = {"status": "eval_failed"}
        else:
            results_summary[name] = {"status": "train_only"}

    total_time = time.time() - t_total

    # Save combined summary
    summary_path = EXPERIMENT_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "configs_run": args.configs,
            "total_time_s": total_time,
            "results": results_summary,
        }, f, indent=2)

    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Summary: {summary_path}")
    for name, res in results_summary.items():
        status = res.get("status", "unknown")
        mse = ""
        if "comparison" in res:
            pct = res["comparison"].get("overall_change_pct", 0)
            mse = f" | MSE change: {pct:+.2f}%"
        print(f"    {name:10s}: {status}{mse}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "from run_adapter_experiment import CONFIGS; print(list(CONFIGS.keys()))"`
Expected: `['base', 'v1', 'v2-prop', 'v2-full']`

**Step 3: Commit**

```bash
git add run_adapter_experiment.py
git commit -m "feat: add experiment runner for 4 adapter configurations"
```

---

## Task 6: adapter_eval.py — Add --baseline_only flag for base config

**Files:**
- Modify: `adapter_eval.py`

**Context:** The experiment runner needs a way to run baseline-only evaluation (no adapter checkpoint). We add a `--baseline_only` flag that skips adapter loading and only evaluates the raw model.

**Step 1: Add baseline-only mode to main()**

In `main()`, after parsing args, add:

```python
    if args.baseline_only:
        # Baseline-only mode: no adapter, just raw OpenVLA MSE
        evaluator = BaselineEvaluator(device=args.device)
        results = evaluator.evaluate(num_episodes=args.num_episodes)
        if args.output_dir:
            # ... save results
        return
```

Actually, it's cleaner to modify `AdapterEvaluator.__init__` to accept `checkpoint_path=None` for baseline mode. Replace the `main()` function entirely:

```python
def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention Adapter")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--baseline_only", action="store_true",
                        help="Evaluate baseline only (no adapter)")
    args = parser.parse_args()

    if args.baseline_only:
        evaluator = AdapterEvaluator(
            checkpoint_path=None, device=args.device, baseline_only=True,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required unless --baseline_only")
        evaluator = AdapterEvaluator(args.checkpoint, device=args.device)

    results = evaluator.evaluate(num_episodes=args.num_episodes)

    if args.output_dir:
        eval_dir = Path(args.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_results = {}
        for k, v in results.items():
            if isinstance(v, dict) and "per_step" in v:
                summary_results[k] = {sk: sv for sk, sv in v.items() if sk != "per_step"}
            else:
                summary_results[k] = v
        out_path = eval_dir / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(summary_results, f, indent=2)
        print(f"Results saved: {out_path}")
```

And modify `__init__` to accept `baseline_only=False`:

In `__init__` signature, change to:
```python
def __init__(self, checkpoint_path: str | None, device: str = "cuda", baseline_only: bool = False):
```

After the boundaries detection block, replace the adapter loading section:
```python
        # ── Load adapter (skip in baseline-only mode) ──
        self.baseline_only = baseline_only
        self.adapter = None
        self.adapter_version = 0
        self.masks_mmap = None

        if not baseline_only and checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            hidden_dim = self.model.config.text_config.hidden_size
            self.adapter_version = ckpt.get("config", {}).get("adapter_version", 1)

            if self.adapter_version == 2:
                self.adapter = AttentionAdapterV2(
                    hidden_dim=hidden_dim,
                    vision_tokens=self.vision_end,
                ).to(device)
            else:
                self.adapter = AttentionAdapter(hidden_dim=hidden_dim).to(device)

            self.adapter.load_state_dict(ckpt["adapter_state_dict"])
            self.adapter.eval()
            print(
                f"Adapter v{self.adapter_version} loaded from {checkpoint_path} "
                f"(step {ckpt.get('global_step', '?')})"
            )

            # Object masks memmap (v2 only)
            if self.adapter_version == 2:
                masks_path = config.DATA_CACHE_DIR / config.SAM_MASKS_FILENAME
                if masks_path.exists():
                    total_steps = np.memmap(
                        str(config.DATA_CACHE_DIR / "images.dat"),
                        dtype=np.uint8, mode="r",
                    ).shape[0] // (256 * 256 * 3)
                    self.masks_mmap = np.memmap(
                        str(masks_path), dtype=np.uint8, mode="r",
                        shape=(total_steps, config.VISION_GRID_SIZE ** 2),
                    )
                    print(f"  Object masks loaded: {masks_path} ({total_steps} steps)")
```

And modify `evaluate()` to handle baseline_only:
```python
        for condition in (["baseline"] if self.baseline_only else ["baseline", "adapter"]):
```

**Step 2: Verify syntax**

Run: `python -c "from adapter_eval import AdapterEvaluator; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add adapter_eval.py
git commit -m "feat(eval): add --baseline_only mode for base config evaluation"
```

---

## Task 7: compare_adapter_results.py — Create results comparison script

**Files:**
- Create: `compare_adapter_results.py`

**Context:** Loads `eval_results.json` from each config's eval directory, produces summary table, bar charts, heatmap, and LaTeX table. Pattern follows `archive/compare_v3_results.py` structure.

**Step 1: Write the comparison script**

Create `compare_adapter_results.py`:

```python
"""Compare adapter experiment results across all configurations.

Loads eval_results.json from each config directory and produces:
1. Summary table (JSON + stdout)
2. Per-dimension MSE bar chart
3. Improvement heatmap
4. LaTeX table for paper

Usage:
    python compare_adapter_results.py
    python compare_adapter_results.py --experiment_dir outputs/experiment_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config

EXPERIMENT_DIR = config.OUTPUT_DIR / "experiment_results"
DIM_NAMES = config.ACTION_DIM_NAMES  # ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_results(experiment_dir: Path) -> dict[str, dict]:
    """Load eval_results.json from each config directory."""
    results = {}
    for config_dir in sorted(experiment_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        eval_path = config_dir / "eval" / "eval_results.json"
        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            results[config_dir.name] = data
            print(f"  Loaded: {config_dir.name} ({eval_path})")
        else:
            print(f"  Skipped: {config_dir.name} (no eval_results.json)")
    return results


def compute_comparison(results: dict[str, dict]) -> dict:
    """Compute comparison statistics relative to baseline."""
    comparison = {}

    base_summary = None
    if "base" in results and "baseline" in results["base"]:
        base_summary = results["base"]["baseline"].get("summary")
    elif "base" in results and "summary" in results["base"]:
        base_summary = results["base"]["summary"]

    if base_summary is None:
        # Use first config's baseline as reference
        for name, data in results.items():
            if "baseline" in data and "summary" in data["baseline"]:
                base_summary = data["baseline"]["summary"]
                break

    if base_summary is None:
        print("WARNING: No baseline found for comparison")
        return comparison

    for name, data in results.items():
        # Get the adapter summary (or baseline for base config)
        if name == "base":
            summary = base_summary
        elif "adapter" in data and "summary" in data["adapter"]:
            summary = data["adapter"]["summary"]
        elif "summary" in data:
            summary = data["summary"]
        else:
            continue

        comparison[name] = {
            "overall_mse": summary["overall_mse"],
            "spatial_mse": summary["spatial_mse"],
            "rotational_mse": summary.get("rotational_mse", 0),
            "per_dim_mse": summary["per_dim_mse"],
            "n_steps": summary.get("n_steps", 0),
        }

        if name != "base":
            base_overall = base_summary["overall_mse"]
            comparison[name]["overall_change_pct"] = (
                (summary["overall_mse"] - base_overall) / base_overall * 100
            )
            comparison[name]["spatial_change_pct"] = (
                (summary["spatial_mse"] - base_summary["spatial_mse"])
                / base_summary["spatial_mse"] * 100
            )
            comparison[name]["per_dim_change_pct"] = {}
            for dim in DIM_NAMES:
                base_dim = base_summary["per_dim_mse"][dim]
                if base_dim > 0:
                    comparison[name]["per_dim_change_pct"][dim] = (
                        (summary["per_dim_mse"][dim] - base_dim) / base_dim * 100
                    )

    return comparison


def print_summary(comparison: dict):
    """Print formatted summary table to stdout."""
    print(f"\n{'=' * 70}")
    print("ADAPTER EXPERIMENT RESULTS COMPARISON")
    print(f"{'=' * 70}")

    # Header
    configs = list(comparison.keys())
    header = f"{'Metric':<20s}"
    for name in configs:
        header += f" {name:>12s}"
    print(header)
    print("-" * 70)

    # Overall MSE
    row = f"{'Overall MSE':<20s}"
    for name in configs:
        row += f" {comparison[name]['overall_mse']:>12.6f}"
    print(row)

    # Spatial MSE
    row = f"{'Spatial MSE':<20s}"
    for name in configs:
        row += f" {comparison[name]['spatial_mse']:>12.6f}"
    print(row)

    # Per-dimension
    print(f"\n{'Per-Dimension MSE':}")
    for dim in DIM_NAMES:
        row = f"  {dim:<18s}"
        for name in configs:
            val = comparison[name]["per_dim_mse"].get(dim, 0)
            row += f" {val:>12.6f}"
        print(row)

    # Change %
    print(f"\n{'Change vs Baseline (%)'}")
    row = f"{'Overall':<20s}"
    for name in configs:
        pct = comparison[name].get("overall_change_pct", 0)
        row += f" {pct:>+11.2f}%"
    print(row)

    for dim in DIM_NAMES:
        row = f"  {dim:<18s}"
        for name in configs:
            pct = comparison[name].get("per_dim_change_pct", {}).get(dim, 0)
            row += f" {pct:>+11.2f}%"
        print(row)

    print(f"{'=' * 70}")


def plot_per_dim_bar(comparison: dict, output_dir: Path):
    """Bar chart of per-dimension MSE for each config."""
    configs = list(comparison.keys())
    n_configs = len(configs)
    n_dims = len(DIM_NAMES)
    x = np.arange(n_dims)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for i, name in enumerate(configs):
        values = [comparison[name]["per_dim_mse"].get(d, 0) for d in DIM_NAMES]
        offset = (i - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i])

    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Dimension MSE by Adapter Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "per_dim_mse_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_improvement_heatmap(comparison: dict, output_dir: Path):
    """Heatmap of per-dimension improvement % vs baseline."""
    configs = [c for c in comparison if c != "base"]
    if not configs:
        return

    n_configs = len(configs)
    data = np.zeros((n_configs, len(DIM_NAMES)))

    for i, name in enumerate(configs):
        for j, dim in enumerate(DIM_NAMES):
            data[i, j] = comparison[name].get("per_dim_change_pct", {}).get(dim, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(abs(data.min()), abs(data.max()), 1)
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_NAMES)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(configs)
    ax.set_title("MSE Change (%) vs Baseline (green = improvement)")

    for i in range(n_configs):
        for j in range(len(DIM_NAMES)):
            ax.text(j, i, f"{data[i, j]:+.1f}%", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Change %")
    plt.tight_layout()
    path = output_dir / "improvement_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_latex_table(comparison: dict, output_dir: Path):
    """Generate LaTeX table for paper inclusion."""
    configs = list(comparison.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adapter experiment results: per-dimension MSE and change vs.\ baseline.}",
        r"\label{tab:adapter-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(configs) + "}",
        r"\toprule",
    ]

    # Header
    header = "Metric"
    for name in configs:
        header += f" & {name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Overall MSE
    row = "Overall MSE"
    for name in configs:
        row += f" & {comparison[name]['overall_mse']:.6f}"
    row += r" \\"
    lines.append(row)

    # Spatial MSE
    row = "Spatial MSE"
    for name in configs:
        row += f" & {comparison[name]['spatial_mse']:.6f}"
    row += r" \\"
    lines.append(row)

    lines.append(r"\midrule")

    # Per-dim
    for dim in DIM_NAMES:
        row = dim
        for name in configs:
            val = comparison[name]["per_dim_mse"].get(dim, 0)
            row += f" & {val:.6f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # Change %
    row = r"$\Delta$ Overall (\%)"
    for name in configs:
        pct = comparison[name].get("overall_change_pct", 0)
        row += f" & {pct:+.2f}"
    row += r" \\"
    lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    path = output_dir / "results_table.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare adapter experiment results")
    parser.add_argument("--experiment_dir", type=str, default=str(EXPERIMENT_DIR))
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    print(f"Loading results from: {experiment_dir}")

    results = load_results(experiment_dir)
    if not results:
        print("No results found!")
        return

    comparison = compute_comparison(results)
    if not comparison:
        print("Could not compute comparison (no baseline)")
        return

    print_summary(comparison)

    # Plots
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_per_dim_bar(comparison, plot_dir)
    plot_improvement_heatmap(comparison, plot_dir)
    generate_latex_table(comparison, plot_dir)

    # Save comparison JSON
    comp_path = experiment_dir / "comparison_summary.json"
    # Convert numpy to native types for JSON
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=to_native)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "from compare_adapter_results import load_results; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add compare_adapter_results.py
git commit -m "feat: add adapter results comparison with plots, heatmap, LaTeX table"
```

---

## Task 8: config.py — Add LoRA constants

**Files:**
- Modify: `config.py:161` (append at end)

**Context:** Add LoRA hyperparameter constants for future `adapter_lora.py`. These are read-only config values — no behavior change yet.

**Step 1: Append LoRA section**

After line 161 (end of file), add:

```python

# ── LoRA Fine-Tuning Infrastructure ──────────────────────────────────
LORA_R = 16                          # LoRA rank
LORA_ALPHA = 32                      # LoRA alpha (scaling = alpha / r)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # LLaMA attention projections
LORA_DROPOUT = 0.05                  # Dropout on LoRA A/B matrices
LORA_LR = 1e-4                      # LoRA learning rate
LORA_MAX_STEPS = 20000               # Max LoRA fine-tuning steps
LORA_WARMUP_STEPS = 200              # Warmup steps for LoRA
LORA_RESULTS_DIR = OUTPUT_DIR / "lora_results"
```

**Step 2: Verify**

Run: `python -c "import config; print(f'LoRA rank={config.LORA_R}, alpha={config.LORA_ALPHA}')" `
Expected: `LoRA rank=16, alpha=32`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat(config): add LoRA hyperparameter constants"
```

---

## Task 9: adapter_lora.py — Create LoRA infrastructure

**Files:**
- Create: `adapter_lora.py`

**Context:** Infrastructure-only: wraps OpenVLA with peft LoRA on LLaMA q_proj/v_proj. Two modes planned: (1) LoRA-only fine-tuning, (2) LoRA+Adapter two-stage. Actual training deferred until adapter v2 results are analyzed.

**Step 1: Write the LoRA infrastructure module**

Create `adapter_lora.py`:

```python
"""LoRA fine-tuning infrastructure for OpenVLA.

Wraps the frozen OpenVLA backbone with LoRA adapters on LLaMA's q_proj and
v_proj layers using the peft library.  Two training modes are supported:

    1. LoRA-only: Fine-tune OpenVLA with LoRA, no attention adapter.
    2. LoRA + Adapter: Two-stage training:
       Stage A — Train attention adapter (LoRA frozen)
       Stage B — Train LoRA (attention adapter frozen)

Infrastructure only — actual training invocation deferred until adapter v2
experiment results are analyzed.

Usage (infrastructure test):
    python adapter_lora.py --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import config

# Lazy peft import to avoid hard dependency
_peft_available = None


def _check_peft():
    global _peft_available
    if _peft_available is None:
        try:
            import peft  # noqa: F401
            _peft_available = True
        except ImportError:
            _peft_available = False
    return _peft_available


def create_lora_model(
    model: nn.Module,
    r: int = config.LORA_R,
    lora_alpha: int = config.LORA_ALPHA,
    target_modules: list[str] = None,
    lora_dropout: float = config.LORA_DROPOUT,
) -> nn.Module:
    """Wrap OpenVLA's LLaMA backbone with LoRA adapters.

    Args:
        model: The full OpenVLA model (must have .language_model).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor (effective scale = alpha / r).
        target_modules: LLaMA submodule names to apply LoRA to.
        lora_dropout: Dropout rate on LoRA A/B matrices.

    Returns:
        peft-wrapped model with LoRA adapters enabled.
    """
    if not _check_peft():
        raise ImportError(
            "peft is required for LoRA. Install with: pip install peft"
        )
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = list(config.LORA_TARGET_MODULES)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Apply to the language model sub-module
    if hasattr(model, "language_model"):
        peft_model = get_peft_model(model.language_model, lora_config)
        model.language_model = peft_model
    else:
        model = get_peft_model(model, lora_config)

    return model


def count_lora_params(model: nn.Module) -> dict:
    """Count trainable (LoRA) vs frozen parameters.

    Returns:
        dict with trainable, frozen, total, and trainable_pct.
    """
    trainable = 0
    frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    total = trainable + frozen
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }


def save_lora_checkpoint(model: nn.Module, path: Path, extra_state: dict = None):
    """Save only the LoRA adapter weights (much smaller than full model).

    Args:
        model: The peft-wrapped model.
        path: Path to save checkpoint.
        extra_state: Additional state (optimizer, step, etc.) to include.
    """
    if not _check_peft():
        raise ImportError("peft required")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract LoRA state dict
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    save_dict = {
        "lora_state_dict": lora_state,
        "lora_config": {
            "r": config.LORA_R,
            "alpha": config.LORA_ALPHA,
            "target_modules": config.LORA_TARGET_MODULES,
            "dropout": config.LORA_DROPOUT,
        },
    }
    if extra_state:
        save_dict.update(extra_state)

    torch.save(save_dict, path)
    print(f"LoRA checkpoint saved: {path} ({len(lora_state)} tensors)")


def load_lora_checkpoint(model: nn.Module, path: str | Path, device: str = "cuda"):
    """Load LoRA adapter weights into a peft-wrapped model.

    Args:
        model: The peft-wrapped model (must already have LoRA applied).
        path: Path to the LoRA checkpoint.
        device: Device to map tensors to.

    Returns:
        extra_state dict (optimizer, step, etc.) if present.
    """
    ckpt = torch.load(path, map_location=device)
    lora_state = ckpt["lora_state_dict"]

    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"  WARNING: LoRA key not found in model: {name}")

    print(f"LoRA checkpoint loaded: {path}")
    return {k: v for k, v in ckpt.items() if k not in ("lora_state_dict", "lora_config")}


def freeze_for_lora_only(model: nn.Module):
    """Freeze all params except LoRA adapters. For mode 1 (LoRA-only)."""
    for param in model.parameters():
        param.requires_grad_(False)
    # Re-enable LoRA params
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def freeze_for_adapter_stage(model: nn.Module, adapter: nn.Module):
    """Freeze model + LoRA, unfreeze adapter. For mode 2, Stage A."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)


def freeze_for_lora_stage(model: nn.Module, adapter: nn.Module):
    """Freeze model + adapter, unfreeze LoRA. For mode 2, Stage B."""
    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def main():
    """Dry-run test: create LoRA model and print param counts."""
    parser = argparse.ArgumentParser(description="LoRA infrastructure test")
    parser.add_argument("--dry_run", action="store_true", help="Test LoRA creation")
    args = parser.parse_args()

    if not args.dry_run:
        print("Use --dry_run to test LoRA infrastructure.")
        print("Actual LoRA training will be added after adapter v2 analysis.")
        return

    if not _check_peft():
        print("peft not installed. Run: pip install peft")
        print("LoRA infrastructure code is ready, but cannot test without peft.")
        return

    print("Loading model for LoRA test...")
    from extract_attention import load_model
    processor, model = load_model(device="cpu")

    # Before LoRA
    before = count_lora_params(model)
    print(f"\nBefore LoRA: {before['total']:,} params (all frozen)")

    # Apply LoRA
    model = create_lora_model(model)

    # After LoRA
    after = count_lora_params(model)
    print(f"After LoRA:  {after['trainable']:,} trainable / {after['total']:,} total")
    print(f"  Trainable: {after['trainable_pct']:.4f}%")
    print(f"  LoRA rank: {config.LORA_R}, alpha: {config.LORA_ALPHA}")
    print(f"  Target modules: {config.LORA_TARGET_MODULES}")

    # Test save/load
    test_path = Path("/tmp/lora_test.pt")
    save_lora_checkpoint(model, test_path, extra_state={"test": True})
    extra = load_lora_checkpoint(model, test_path)
    print(f"  Save/load test: {'PASS' if extra.get('test') else 'FAIL'}")

    print("\nLoRA infrastructure ready.")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "from adapter_lora import create_lora_model, count_lora_params; print('OK')"`
Expected: `OK` (or ImportError for peft, which is expected — the lazy import handles this)

**Step 3: Commit**

```bash
git add adapter_lora.py
git commit -m "feat: add LoRA fine-tuning infrastructure (peft wrapper, save/load, freeze modes)"
```

---

## Execution Order & Dependencies

```
Task 1: adapter_eval.py auto-detect (no dependencies)
Task 2: adapter_eval.py get_v3_ctx_for_eval upgrade (depends on Task 1)
Task 3: adapter_eval.py object mask loading (depends on Task 2)
Task 4: adapter_train.py flags (independent of Tasks 1-3)
Task 5: run_adapter_experiment.py (depends on Tasks 1-4 conceptually, but syntactically independent)
Task 6: adapter_eval.py baseline_only (depends on Task 3)
Task 7: compare_adapter_results.py (independent, creates new file)
Task 8: config.py LoRA constants (independent)
Task 9: adapter_lora.py (depends on Task 8)
```

**Parallelizable groups:**
- Tasks 1-3 (adapter_eval.py) must be sequential
- Task 4 (adapter_train.py) can run parallel with Tasks 1-3
- Tasks 7, 8 are independent
- Task 9 depends on Task 8

---

## Post-Implementation: Running the Experiment

After all tasks are complete and SAM preprocessing finishes:

```bash
# Full experiment (4 configs, 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_adapter_experiment.py --gpus 0,1,2,3

# Compare results
python compare_adapter_results.py

# Check results
cat outputs/experiment_results/comparison_summary.json
```

---

## Notes

- **No formal test suite**: This project has no `tests/` directory. Verification is via import checks and manual runtime testing. The `--dry_run` pattern in `adapter_lora.py` is the closest to a test.
- **SAM dependency**: Tasks 1-9 can all be implemented now. Only the actual training (Phase 3) requires SAM preprocessing to complete for v2-full config.
- **Reference file**: `archive/compare_v3_results.py` was used as the template for `compare_adapter_results.py` plotting patterns.
