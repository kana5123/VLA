# Cross-Model Object-Centric Adapter Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 4개 VLA 모델(OpenVLA, ECoT, SpatialVLA, TraceVLA)에 V2-Full object-centric adapter를 학습/평가하여, base model 대비 성능 차이를 측정한다.

**Architecture:** 기존 OpenVLA 전용 코드를 `model_registry`를 활용하여 multi-model 지원으로 일반화. 각 모델별 adapter(hidden_dim, num_heads, vision_tokens 다름)를 독립 학습. attention 패칭은 아키텍처별 분기(LLaMA/Gemma2/Phi3).

**Tech Stack:** PyTorch 2.4, Accelerate (multi-GPU DDP), transformers 4.57.6 (interp env), model_registry.py, Bridge V2 dataset (53K episodes), SAM2+GroundingDINO masks

---

## Pre-Execution Review & Fixes Applied (2026-02-22)

코드 리뷰에서 발견된 Critical/High 이슈들이 plan 실행 전에 수정됨:

### Already Completed (코드에 반영됨)

| Task | Status | 내용 |
|------|--------|------|
| Task 1 | **DONE** | `model_registry.py`: action_type, source_layer, target_layers, auto_model_class, get_adapter_config() 추가 |
| Task 2 | **DONE** | `adapter_model.py`: vision_tokens 파라미터 추가 (H1 fix) |
| Task 4 | **DONE** (다른 방식) | `attention_v3.py`: Gemma2/Phi3V patched forward 함수 직접 구현 (hook 방식 대신) |
| C1 fix | **DONE** | `attention_v3.py:379,394,404`: text-sink query index `[b,h,0,...]` → `[b,h,-1,...]` |
| C5 fix | **DONE** | `adapter_data.py`: `resize_mask()` 유틸리티 추가 (TraceVLA 313 tokens 호환) |

### Critical Discoveries (plan 코드 수정 필요)

| 발견 | 영향 | 수정 방향 |
|------|------|----------|
| **SpatialVLA vision_tokens = 256** (not 196) | registry 수정 완료 | ✅ 반영됨 |
| **TraceVLA vision_tokens = 313** (not 196/144) | registry 수정 완료 | ✅ 반영됨 |
| **TraceVLA는 `AutoModelForCausalLM`** (not Vision2Seq) | Task 3 코드 수정 필요 | auto_model_class="AutoModelForCausalLM" in registry |
| **SpatialVLA는 `AutoModel`** + monkey-patch 필요 | Task 3 코드 수정 필요 | auto_model_class="AutoModel", cross_model_extract.py 참고 |
| **SAM masks = 256 tokens** | TraceVLA(313)에 resize_mask 필요 | adapter_data.py에 resize_mask() 추가됨 |
| **Python env: `/home/kana5123/miniconda3/envs/interp/bin/python`** | 모든 실행에 이 env 사용 | transformers 4.57.6 + torch 2.4.0 |

### Remaining Tasks (plan 실행 시 수행)

- **Task 3**: `load_model_from_registry` — TraceVLA=AutoModelForCausalLM, SpatialVLA=AutoModel+monkey-patch 반영
- **Task 5**: `adapter_train.py` multi-model 지원 — config 상수 제거, model_cfg 사용
- **Task 5b**: TraceVLA continuous action loss 구현
- **Task 6**: `adapter_eval.py` multi-model 지원
- **Task 7-15**: 실행 태스크

---

### Task 1: model_registry.py에 action_type 필드 추가

**Files:**
- Modify: `model_registry.py:14-35`

**Step 1: VLAModelConfig에 action_type, source_layer, target_layers 필드 추가**

```python
@dataclass
class VLAModelConfig:
    # ... existing fields ...
    action_type: str = "discrete"     # "discrete" (CE loss) or "continuous" (MSE loss)
    source_layer: int = -5            # Hidden state capture layer (relative to num_layers if negative)
    target_layers: list[int] = field(default_factory=list)  # VAR-applied layers (empty = last 4)
```

**Step 2: 각 모델 registration에 값 지정**

```python
# OpenVLA:
action_type="discrete", source_layer=27, target_layers=[28, 29, 30, 31]

# ECoT:
action_type="discrete", source_layer=27, target_layers=[28, 29, 30, 31]

# SpatialVLA:
action_type="discrete", source_layer=21, target_layers=[22, 23, 24, 25]

# TraceVLA:
action_type="continuous", source_layer=27, target_layers=[28, 29, 30, 31]
```

**Step 3: helper 메서드 추가**

```python
def get_adapter_config(self) -> dict:
    """Return adapter-relevant parameters from model config."""
    tl = self.target_layers if self.target_layers else list(range(self.num_layers - 4, self.num_layers))
    sl = self.source_layer if self.source_layer >= 0 else self.num_layers + self.source_layer
    return {
        "hidden_dim": self.hidden_dim,
        "num_heads": self.num_heads,
        "num_target_layers": len(tl),
        "target_layers": tl,
        "source_layer": sl,
        "vision_tokens": self.num_vision_tokens,
        "action_type": self.action_type,
    }
```

**Step 4: 확인**

Run: `python -c "from model_registry import get_model; c = get_model('tracevla-phi3v'); print(c.action_type, c.source_layer, c.target_layers)"`
Expected: `continuous 27 [28, 29, 30, 31]`

**Step 5: Commit**

```bash
git add model_registry.py
git commit -m "feat(registry): add action_type, source_layer, target_layers to VLAModelConfig"
```

---

### Task 2: adapter_model.py — vision_tokens를 생성자 파라미터로 추가

**Files:**
- Modify: `adapter_model.py:203-232`

**Step 1: AttentionAdapterV2 __init__ 시그니처에 vision_tokens 추가**

현재 `adapter_model.py:231`에서 `vision_tokens = config.VISION_GRID_SIZE ** 2` 로 하드코딩됨.

```python
class AttentionAdapterV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_target_layers: int = config.ADAPTER_NUM_TARGET_LAYERS,
        num_heads: int = config.NUM_HEADS,
        intermediate_dim: int = config.ADAPTER_INTERMEDIATE_DIM,
        mask_dim: int = config.ADAPTER_V2_MASK_DIM,
        query_dim: int = config.ADAPTER_V2_QUERY_DIM,
        temperature: float = config.ADAPTER_V2_TEMPERATURE,
        blend_init: float = config.ADAPTER_V2_BLEND_INIT,
        dropout: float = config.ADAPTER_DROPOUT,
        vision_tokens: int | None = None,    # NEW: 모델별 vision token 수
    ):
        super().__init__()
        # ...
        # 변경: 하드코딩 제거
        if vision_tokens is None:
            vision_tokens = config.VISION_GRID_SIZE ** 2  # backward compat
        self.vision_tokens = vision_tokens
        self.mask_linear = nn.Linear(vision_tokens, mask_dim)
```

**Step 2: 확인**

Run: `python -c "from adapter_model import AttentionAdapterV2; a = AttentionAdapterV2(hidden_dim=2304, num_heads=8, num_target_layers=4, vision_tokens=144); print(a.mask_linear, a.param_count())"`
Expected: `Linear(in_features=144, out_features=64, bias=True) ...` (reduced param count)

**Step 3: Commit**

```bash
git add adapter_model.py
git commit -m "feat(adapter): parameterize vision_tokens in AttentionAdapterV2"
```

---

### Task 3: extract_attention.py — 범용 모델 로딩 함수 추가

**Files:**
- Modify: `extract_attention.py` (new function `load_model_from_registry`)
- Reference: `cross_model_extract.py:59-130` (기존 `load_vla_model` 참고)

**Step 1: load_model_from_registry 함수 추가**

`extract_attention.py`에 `load_model` 기존 함수 아래에 추가.

**중요**: 모델별 auto_model_class가 다름 — `model_registry.py`의 `auto_model_class` 필드 사용.
- OpenVLA/ECoT: `AutoModelForVision2Seq` (기본값)
- SpatialVLA: `AutoModel` + `_validate_images_text_input_order` monkey-patch 필요
- TraceVLA: `AutoModelForCausalLM` (Phi3VConfig이 Vision2Seq에 등록 안됨)

참고 코드: `cross_model_extract.py:25-44` (monkey-patch), `cross_model_extract.py:59-130` (load_vla_model)

```python
def load_model_from_registry(model_name: str, device: str = "cuda"):
    """Load any VLA model using registry config."""
    from model_registry import get_model
    model_cfg = get_model(model_name)

    # Apply monkey-patches for SpatialVLA processor
    if model_cfg.architecture == "gemma2":
        import transformers.processing_utils as _proc_utils
        if not hasattr(_proc_utils, '_validate_images_text_input_order'):
            from transformers.image_utils import is_valid_image
            def _validate(images, text):
                if images is not None and not is_valid_image(images):
                    if isinstance(images, str):
                        images, text = text, images
                return images, text
            _proc_utils._validate_images_text_input_order = _validate

    processor = AutoProcessor.from_pretrained(
        model_cfg.hf_id, trust_remote_code=model_cfg.trust_remote_code
    )

    # Select auto_model_class from registry
    auto_cls_name = model_cfg.auto_model_class
    if auto_cls_name == "AutoModelForVision2Seq":
        from transformers import AutoModelForVision2Seq as AutoCls
    elif auto_cls_name == "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM as AutoCls
    elif auto_cls_name == "AutoModel":
        from transformers import AutoModel as AutoCls
    else:
        raise ValueError(f"Unknown auto_model_class: {auto_cls_name}")

    model = AutoCls.from_pretrained(
        model_cfg.hf_id,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=model_cfg.attn_impl,
    ).to(device).eval()

    print(f"Loaded {model_cfg.name}: {model_cfg.num_layers}L x {model_cfg.num_heads}H, hidden={model_cfg.hidden_dim}")
    return processor, model, model_cfg
```

**Step 2: get_layers 유틸리티 함수 추가**

```python
def get_layers(model, model_cfg):
    """Navigate to transformer layers using model_cfg.layers_path."""
    obj = model
    for attr in model_cfg.layers_path.split("."):
        obj = getattr(obj, attr)
    return obj
```

**Step 3: 확인**

Run: `python -c "from extract_attention import load_model_from_registry; print('OK')"`
Expected: `OK` (import 성공)

**Step 4: Commit**

```bash
git add extract_attention.py
git commit -m "feat(extract): add load_model_from_registry for multi-model support"
```

---

### Task 4: attention_v3.py — 아키텍처별 attention 패칭 일반화 ✅ COMPLETED

**Status:** Pre-execution에서 완료됨. 아래 plan 코드와 다른 방식으로 구현됨.

**실제 구현 (attention_v3.py에 적용됨):**
1. `_apply_v3_enhancements(ctx, attn_weights, layer_idx)` — VAR/ACT/VTR/BG 공통 로직 추출
2. `_apply_v3_spin(ctx, attn_output, attn_weights, layer_idx)` — SPIN 공통 로직 추출
3. `_make_v3_patched_forward_gemma2(ctx)` — Gemma2 전용 (softcap, GQA)
4. `_make_v3_patched_forward_phi3v(ctx)` — Phi3V 전용 (fused qkv_proj)
5. `install_v3_patch(ctx, architecture="llama", model=None)` — 아키텍처 파라미터 추가
6. `uninstall_v3_patch()` — 어떤 아키텍처든 복원

**원래 계획의 hook-based 방식 대신 직접 forward 구현을 선택한 이유:**
- Hook 방식은 `attn_output` 재계산 필요 (비효율적)
- 아키텍처별 forward 직접 구현이 post-softmax 주입에 정확하고 효율적

**Files:**
- Modify: `attention_v3.py:615-764`

현재 `install_v3_patch`는 `LlamaAttention.forward`만 패칭. SpatialVLA(Gemma2)와 TraceVLA(Phi3V)도 패칭해야 함.

**Step 1: _make_generic_patched_forward 팩토리 함수 추가**

기존 `_make_v3_patched_forward`는 LlamaAttention 전용(apply_rotary_pos_emb, repeat_kv import). 각 아키텍처별 forward를 생성하는 팩토리가 필요.

핵심 아이디어: **모든 아키텍처의 attention forward는 결국 `attn_weights = softmax(QK^T/sqrt(d))` 직후에 manipulation 가능.** 기존 forward를 래핑하는 post-hook 방식으로 통일.

```python
_original_fns_v3: dict[type, Callable] = {}  # 아키텍처별 원본 forward 저장


def install_v3_patch_for_arch(ctx: V3Context, architecture: str) -> None:
    """Patch attention forward for the given architecture."""
    global _original_fns_v3

    if architecture in ("llama",):
        from transformers.models.llama.modeling_llama import LlamaAttention
        attn_cls = LlamaAttention
    elif architecture == "gemma2":
        # SpatialVLA uses trust_remote_code → Gemma2Attention from remote module
        # We need to patch the class AFTER model loading
        # Use register_forward_hook approach instead
        return _install_hook_based_patch(ctx, architecture)
    elif architecture == "phi3_v":
        # Same: trust_remote_code → use hook approach
        return _install_hook_based_patch(ctx, architecture)
    else:
        raise ValueError(f"Unsupported architecture for patching: {architecture}")

    if attn_cls in _original_fns_v3:
        return  # Already patched

    _original_fns_v3[attn_cls] = attn_cls.forward
    attn_cls.forward = _make_v3_patched_forward(ctx)
    print(f"[attention_v3] Patched {attn_cls.__name__}.forward")
```

**Step 2: Hook-based 패칭 (Gemma2, Phi3)**

SpatialVLA/TraceVLA는 `trust_remote_code=True`로 로드되므로 모듈 클래스가 동적. 대안: 모델 로드 후 개별 레이어의 `self_attn` 모듈에 `register_forward_hook` 사용.

```python
_v3_hooks: list = []  # installed hooks for cleanup


def _install_hook_based_patch(ctx: V3Context, architecture: str) -> None:
    """Install post-softmax attention manipulation via forward hooks.

    For architectures with trust_remote_code (Gemma2, Phi3), we can't
    import and monkey-patch the attention class directly. Instead we
    register forward hooks on each attention layer's self_attn module.
    """
    # Hooks are registered in adapter_train.py AFTER model loading,
    # when we have access to the actual model object.
    # This function just sets a flag.
    ctx._use_hook_based_patch = True
    ctx._hook_architecture = architecture
    print(f"[attention_v3] Hook-based patch mode for {architecture}")


def install_hooks_on_model(model, model_cfg, ctx: V3Context) -> None:
    """Register forward hooks on attention layers for VAR injection.

    Called after model loading with the actual model object.
    Works for any architecture by hooking into output_attentions or
    by wrapping the attention module's forward to inject post-softmax manipulation.
    """
    global _v3_hooks
    from extract_attention import get_layers

    layers = get_layers(model, model_cfg)
    target_layer_set = set(model_cfg.target_layers or
                           list(range(model_cfg.num_layers - 4, model_cfg.num_layers)))

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in target_layer_set:
            continue

        attn_module = getattr(layer, model_cfg.attn_module, None)
        if attn_module is None:
            continue

        # Wrap the attention module's forward to intercept post-softmax
        original_forward = attn_module.forward
        hook = _make_attn_hook(ctx, layer_idx, original_forward, attn_module)
        attn_module.forward = hook
        _v3_hooks.append((attn_module, original_forward))

    print(f"[attention_v3] Installed hooks on {len(_v3_hooks)} attention layers")


def _make_attn_hook(ctx, layer_idx, original_forward, attn_module):
    """Create a wrapped forward that applies VAR after softmax.

    This replaces the attention module's forward entirely, calling the
    original forward with output_attentions=True to get attention weights,
    then applying VAR manipulation and recomputing the output.
    """
    def hooked_forward(*args, **kwargs):
        # Force output_attentions to get weights
        kwargs['output_attentions'] = True
        result = original_forward(*args, **kwargs)

        attn_output, attn_weights_raw = result[0], result[1]

        if ctx.is_active(layer_idx) and attn_weights_raw is not None:
            # Apply VAR on attention weights
            attn_weights_modified = attn_weights_raw.clone()

            if ctx.use_var:
                if ctx.dynamic_sink_detection:
                    last_attn = attn_weights_modified[:, :, -1, :]
                    sink_indices = detect_sinks(last_attn, alpha=ctx.sink_alpha)
                    if not sink_indices:
                        sink_indices = ctx.var_sink_indices
                else:
                    sink_indices = ctx.var_sink_indices

                attn_weights_modified = apply_var(
                    attn_weights_modified, sink_indices, ctx.vision_end,
                    ctx.effective_var_p(), ctx.var_rho,
                    per_head_p=ctx.get_per_head_p(layer_idx),
                    redistribution_weights=ctx.redistribution_weights,
                    text_sink_enabled=ctx.text_sink_enabled,
                    text_sink_p=ctx.text_sink_p,
                    text_sink_threshold=ctx.text_sink_threshold,
                    text_end=ctx.text_end,
                )

            # Recompute attn_output with modified weights
            # Get value_states from the module (architecture-specific)
            # This is complex — simpler approach: use the attention hook
            # at the softmax output level.
            # ALTERNATIVE: Use the existing _make_v3_patched_forward pattern
            # but for each architecture.

            # For now: return modified result
            return (attn_output,) + result[1:]  # TODO: recompute with modified weights

        return result
    return hooked_forward


def uninstall_hooks() -> None:
    """Restore all hooked attention modules to original forward."""
    global _v3_hooks
    for attn_module, original_forward in _v3_hooks:
        attn_module.forward = original_forward
    print(f"[attention_v3] Uninstalled {len(_v3_hooks)} hooks")
    _v3_hooks.clear()
```

**중요 결정점**: Hook-based 방식은 `attn_output`을 재계산해야 하는 문제가 있음. 대안으로 아키텍처별 `_make_v3_patched_forward_gemma2` / `_make_v3_patched_forward_phi3v` 를 개별 작성하는 것이 더 안정적.

**Step 3: SpatialVLA (Gemma2) 패칭 함수**

SpatialVLA는 `trust_remote_code=True`이므로 모델 로드 후 attention 클래스를 동적으로 가져와야 함:

```python
def install_v3_patch_gemma2(ctx: V3Context, model) -> None:
    """Patch Gemma2 attention for SpatialVLA."""
    global _original_fns_v3

    # Get the attention class from the loaded model
    from extract_attention import get_layers
    from model_registry import get_model
    model_cfg = get_model("spatialvla-4b")
    layers = get_layers(model, model_cfg)
    attn_module = getattr(layers[0], model_cfg.attn_module)
    attn_cls = type(attn_module)

    if attn_cls in _original_fns_v3:
        return

    _original_fns_v3[attn_cls] = attn_cls.forward
    attn_cls.forward = _make_v3_patched_forward_generic(ctx, attn_cls)
    print(f"[attention_v3] Patched {attn_cls.__name__}.forward")
```

**Step 4: install_v3_patch 확장**

기존 `install_v3_patch`에 architecture 파라미터 추가 (backward compatible):

```python
def install_v3_patch(ctx: Optional[V3Context] = None, architecture: str = "llama", model=None) -> None:
    """Patch attention forward with V3-enhanced version.

    Args:
        ctx: V3Context configuration
        architecture: "llama", "gemma2", or "phi3_v"
        model: The loaded model (required for gemma2/phi3_v to get attention class)
    """
    if architecture == "llama":
        # Existing LlamaAttention patching (unchanged)
        _install_llama_patch(ctx)
    elif architecture in ("gemma2", "phi3_v"):
        if model is None:
            raise ValueError(f"model parameter required for {architecture} patching")
        _install_generic_patch(ctx, model, architecture)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
```

**Step 5: uninstall_v3_patch 확장**

```python
def uninstall_v3_patch() -> None:
    """Restore all patched attention forwards."""
    global _original_fns_v3, _original_fn_v3
    # Existing LlamaAttention restore
    if _original_fn_v3 is not None:
        from transformers.models.llama.modeling_llama import LlamaAttention
        LlamaAttention.forward = _original_fn_v3
        _original_fn_v3 = None
    # Generic restores
    for attn_cls, original_fn in _original_fns_v3.items():
        attn_cls.forward = original_fn
    _original_fns_v3.clear()
    print("[attention_v3] All patches uninstalled.")
```

**Step 6: 확인**

Run: `python -c "from attention_v3 import install_v3_patch; print('import OK')"`
Expected: `import OK`

**Step 7: Commit**

```bash
git add attention_v3.py
git commit -m "feat(attn): generalize attention patching for Gemma2 and Phi3V architectures"
```

---

### Task 5: adapter_train.py — multi-model 지원

**Files:**
- Modify: `adapter_train.py:410-742`

**Step 1: --model 인자 추가**

```python
parser.add_argument("--model", type=str, default="openvla-7b",
                    choices=["openvla-7b", "ecot-7b", "spatialvla-4b", "tracevla-phi3v"],
                    help="VLA model from model_registry")
```

**Step 2: 모델 로딩 일반화**

기존 `load_model()` → `load_model_from_registry(args.model)`:

```python
from extract_attention import load_model_from_registry, get_layers
from model_registry import get_model

model_cfg = get_model(args.model)
processor, model, model_cfg = load_model_from_registry(args.model, device=str(device))
model.eval()
for param in model.parameters():
    param.requires_grad_(False)

# Adapter config from model
adapter_cfg = model_cfg.get_adapter_config()  # NEW: helper from Task 1
```

**Step 3: Token boundary 탐지 일반화**

기존 `detect_token_boundaries`는 OpenVLA 전용. 모델별 prompt template 사용:

```python
from PIL import Image as PILImage
dummy_image = PILImage.new("RGB", (256, 256), color=(128, 128, 128))
prompt = model_cfg.prompt_template.format(instruction="pick up the object")
inputs = processor(prompt, dummy_image, return_tensors="pt").to(device)

# Vision tokens: model_cfg.num_vision_tokens
vision_end = model_cfg.num_vision_tokens
# Text end: total tokens - action tokens (approximate)
total_tokens = inputs["input_ids"].shape[-1]
text_end = total_tokens  # action tokens haven't been generated yet
```

**Step 4: 어댑터 생성 시 모델별 파라미터 사용**

```python
if args.adapter_version == 2:
    adapter = AttentionAdapterV2(
        hidden_dim=adapter_cfg["hidden_dim"],
        num_target_layers=adapter_cfg["num_target_layers"],
        num_heads=adapter_cfg["num_heads"],
        vision_tokens=adapter_cfg["vision_tokens"],
    )
else:
    adapter = AttentionAdapter(
        hidden_dim=adapter_cfg["hidden_dim"],
        num_target_layers=adapter_cfg["num_target_layers"],
        num_heads=adapter_cfg["num_heads"],
    )
```

**Step 5: Layer 접근 일반화**

`forward_with_adapter`와 `evaluate` 함수에서 layer 접근 패턴 변경:

기존:
```python
if hasattr(model, "language_model"):
    layers = model.language_model.model.layers
else:
    layers = model.model.layers
hook = layers[config.ADAPTER_SOURCE_LAYER].register_forward_hook(capture_hook)
```

변경:
```python
layers = get_layers(model, model_cfg)
hook = layers[adapter_cfg["source_layer"]].register_forward_hook(capture_hook)
```

**Step 6: config 상수 → model_cfg 대체**

`forward_with_adapter`와 `evaluate`에서 다음 상수들을 `adapter_cfg`/`model_cfg` 값으로 대체:
- `config.NUM_LAYERS` → `model_cfg.num_layers`
- `config.NUM_HEADS` → `model_cfg.num_heads`
- `config.ADAPTER_SOURCE_LAYER` → `adapter_cfg["source_layer"]`
- `config.ADAPTER_TARGET_LAYERS` → `adapter_cfg["target_layers"]`
- `config.NUM_ACTION_TOKENS` → `model_cfg.action_tokens`

**Step 7: Loss 분기 (discrete vs continuous)**

기존 (discrete only):
```python
loss_i = F.cross_entropy(logits.float(), target)
```

변경:
```python
if adapter_cfg["action_type"] == "discrete":
    target = torch.tensor([target_token_ids[token_idx]], device=device, dtype=torch.long)
    loss_i = F.cross_entropy(logits.float(), target)
elif adapter_cfg["action_type"] == "continuous":
    # TraceVLA: continuous action prediction
    # logits의 마지막 hidden state에서 action head를 통해 연속값 출력
    # model-specific: TraceVLA의 action head 접근 방법 확인 필요
    predicted = logits.float()  # 또는 model의 action_head 출력
    target_val = torch.tensor([gt_action[token_idx]], device=device, dtype=torch.float32)
    loss_i = F.mse_loss(predicted, target_val)
```

**주의**: TraceVLA의 action 출력 방식은 별도 확인 필요 (Task 5b에서 상세화).

**Step 8: V3 Context 일반화**

```python
ctx = V3Context(
    active=True,
    use_var=True,
    var_p=config.VAR_P,
    var_rho=config.VAR_RHO,
    var_sink_indices=list(config.VAR_SINK_INDICES),
    dynamic_sink_detection=config.DYNAMIC_SINK_DETECTION,
    sink_alpha=config.SINK_ALPHA,
    vision_end=adapter_cfg["vision_tokens"],  # model-specific
    enhancement_layers=set(adapter_cfg["target_layers"]),  # model-specific
    text_end=text_end,
    text_sink_enabled=config.VAR_TEXT_SINK_ENABLED,
    text_sink_p=config.VAR_TEXT_SINK_P,
    text_sink_threshold=config.VAR_TEXT_SINK_THRESHOLD,
)

# Architecture-specific patching
install_v3_patch(ctx, architecture=model_cfg.architecture, model=model)
```

**Step 9: Checkpoint에 모델 정보 저장**

```python
torch.save({
    "adapter_state_dict": raw_adapter.state_dict(),
    # ... existing fields ...
    "config": {
        "model_name": args.model,
        "architecture": model_cfg.architecture,
        "hidden_dim": adapter_cfg["hidden_dim"],
        "num_heads": adapter_cfg["num_heads"],
        "num_target_layers": adapter_cfg["num_target_layers"],
        "target_layers": adapter_cfg["target_layers"],
        "source_layer": adapter_cfg["source_layer"],
        "vision_tokens": adapter_cfg["vision_tokens"],
        "action_type": adapter_cfg["action_type"],
        # ... other existing fields ...
    },
}, path)
```

**Step 10: 확인**

Run: `python adapter_train.py --model openvla-7b --help`
Expected: `--model` 옵션이 도움말에 표시

**Step 11: Commit**

```bash
git add adapter_train.py
git commit -m "feat(train): multi-model adapter training with model_registry"
```

---

### Task 5b: TraceVLA continuous action loss 구현

**Files:**
- Modify: `adapter_train.py` (Task 5의 loss 분기 상세화)
- Reference: TraceVLA 소스 코드에서 action head 구조 확인

TraceVLA는 autoregressive action token이 아닌 continuous 7-dim 출력. 학습 시 CE loss 대신 MSE loss 사용.

**Step 1: TraceVLA action head 확인**

Run: `python -c "from transformers import AutoModelForVision2Seq; m = AutoModelForVision2Seq.from_pretrained('furonghuang-lab/tracevla_phi3v', trust_remote_code=True, attn_implementation='eager'); print(type(m)); print([n for n, _ in m.named_modules() if 'action' in n.lower() or 'head' in n.lower()])"`

Expected: action head의 구조와 이름 확인

**Step 2: forward_with_adapter의 TraceVLA 분기 구현**

TraceVLA가 discrete tokenization을 사용하지 않으므로:
- `ActionTokenizer` 대신 raw continuous action 사용
- Teacher forcing 시 ground-truth action을 직접 제공
- Loss는 MSE

```python
if adapter_cfg["action_type"] == "continuous":
    # TraceVLA: single forward pass, MSE loss
    outputs = model(**model_inputs, use_cache=False)
    # Extract action predictions (model-specific)
    pred_actions = extract_continuous_actions(model, outputs, model_cfg)
    gt_actions = torch.tensor(gt_action, device=device, dtype=torch.float32)
    total_loss = F.mse_loss(pred_actions, gt_actions)
```

**주의**: 이 태스크는 TraceVLA 모델 구조를 실제로 확인한 후에 정확한 구현이 가능. 만약 TraceVLA도 내부적으로 discrete tokenization을 사용한다면 (많은 VLA가 그렇듯) CE loss를 그대로 사용할 수 있음.

**Step 3: Commit**

```bash
git add adapter_train.py
git commit -m "feat(train): add continuous action loss for TraceVLA"
```

---

### Task 6: adapter_eval.py — multi-model 평가 지원

**Files:**
- Modify: `adapter_eval.py:35-388`

**Step 1: --model 인자 추가**

```python
parser.add_argument("--model", type=str, default="openvla-7b",
                    choices=["openvla-7b", "ecot-7b", "spatialvla-4b", "tracevla-phi3v"])
```

**Step 2: AdapterEvaluator.__init__ 일반화**

```python
class AdapterEvaluator:
    def __init__(
        self,
        checkpoint_path: str | None,
        model_name: str = "openvla-7b",
        device: str = "cuda",
        baseline_only: bool = False,
    ):
        from extract_attention import load_model_from_registry
        from model_registry import get_model

        self.model_cfg = get_model(model_name)
        self.adapter_cfg = self.model_cfg.get_adapter_config()

        print(f"Loading {model_name}...")
        self.processor, self.model, _ = load_model_from_registry(model_name, device=device)
        self.model.eval()

        # Token boundaries
        self.vision_end = self.model_cfg.num_vision_tokens
        # ... rest similar but using model_cfg ...
```

**Step 3: Layer 접근 일반화**

`get_v3_ctx_for_eval`에서 `config.ADAPTER_SOURCE_LAYER` → `self.adapter_cfg["source_layer"]`, 기타 config 상수 대체.

**Step 4: 출력 경로에 모델명 포함**

```python
eval_dir = Path(args.output_dir) if args.output_dir else config.ADAPTER_RESULTS_DIR / "eval" / model_name
```

**Step 5: 확인**

Run: `python adapter_eval.py --model openvla-7b --baseline_only --help`
Expected: `--model` 옵션 표시

**Step 6: Commit**

```bash
git add adapter_eval.py
git commit -m "feat(eval): multi-model adapter evaluation support"
```

---

### Task 7: run_adapter_experiment.py — base + v2-full only

**Files:**
- Modify: `run_adapter_experiment.py:31-52`

**Step 1: CONFIGS를 base + v2-full만 남김**

```python
CONFIGS = {
    "base": {
        "skip_training": True,
        "adapter_version": None,
        "description": "Raw VLA baseline (no adapter)",
    },
    "v2-full": {
        "adapter_version": 2,
        "freeze_blend": False,
        "description": "AttentionAdapterV2, object-centric dynamic adapter",
    },
}
```

**Step 2: --model 인자 추가**

```python
parser.add_argument("--model", type=str, default="openvla-7b",
                    help="VLA model name from model_registry")
```

**Step 3: 확인**

Run: `python -c "from run_adapter_experiment import CONFIGS; print(list(CONFIGS.keys()))"`
Expected: `['base', 'v2-full']`

**Step 4: Commit**

```bash
git add run_adapter_experiment.py
git commit -m "refactor(experiment): reduce configs to base + v2-full only, add --model arg"
```

---

### Task 8: V1 학습 프로세스 중단 및 GPU 확보

**Files:**
- 없음 (프로세스 관리)

**Step 1: 현재 V1 프로세스 확인**

Run: `ps aux | grep adapter_train | grep -v grep`
Expected: GPU 1-4에서 실행 중인 V1 프로세스 PID 확인

**Step 2: V1 프로세스 종료**

```bash
# accelerate launch의 메인 프로세스 그룹 종료
kill <PIDs from Step 1>
```

**Step 3: GPU 해제 확인**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
Expected: GPUs 1-4 메모리 사용량 < 100MB

**Step 4: V1 체크포인트 보존 확인**

Run: `ls outputs/experiment_results/v1/checkpoints/ 2>/dev/null || echo "V1 dir not found"`

---

### Task 9: OpenVLA V2-Full 어댑터 학습 실행

**Files:**
- 없음 (실행만)

**Step 1: 학습 시작**

```bash
cd /home/kana5123/ATLASVLA
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    adapter_train.py \
    --model openvla-7b \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/openvla-7b/v2-full \
    2>&1 | tee outputs/experiment_results/openvla-7b/v2-full/train.log
```

Expected:
- 학습 시작, loss 출력
- `best.pt` 체크포인트 생성
- 예상 소요: 6-12시간

**Step 2: 학습 모니터링**

```bash
tail -20 outputs/experiment_results/openvla-7b/v2-full/train.log
```

모니터링 지표:
- Train loss 감소 추세
- MeanP: 0.05-0.30 범위
- BlendAlpha: 0에서 서서히 증가
- Val loss: 감소 추세

---

### Task 10: Baseline 평가 (4개 모델)

**Files:**
- 없음 (실행만)

각 모델의 base (어댑터 없음) 성능을 test set에서 측정.

**Step 1: OpenVLA baseline 평가**

```bash
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --model openvla-7b \
    --baseline_only \
    --output_dir outputs/experiment_results/openvla-7b/base/eval \
    2>&1 | tee outputs/experiment_results/openvla-7b/base/eval.log
```

**Step 2: ECoT baseline 평가**

```bash
CUDA_VISIBLE_DEVICES=6 python adapter_eval.py \
    --model ecot-7b \
    --baseline_only \
    --output_dir outputs/experiment_results/ecot-7b/base/eval \
    2>&1 | tee outputs/experiment_results/ecot-7b/base/eval.log
```

**Step 3: SpatialVLA baseline 평가**

```bash
CUDA_VISIBLE_DEVICES=7 python adapter_eval.py \
    --model spatialvla-4b \
    --baseline_only \
    --output_dir outputs/experiment_results/spatialvla-4b/base/eval \
    2>&1 | tee outputs/experiment_results/spatialvla-4b/base/eval.log
```

**Step 4: TraceVLA baseline 평가** (SpatialVLA 완료 후 GPU 7 재사용)

```bash
CUDA_VISIBLE_DEVICES=7 python adapter_eval.py \
    --model tracevla-phi3v \
    --baseline_only \
    --output_dir outputs/experiment_results/tracevla-phi3v/base/eval \
    2>&1 | tee outputs/experiment_results/tracevla-phi3v/base/eval.log
```

---

### Task 11: ECoT V2-Full 어댑터 학습 (OpenVLA 학습 완료 후)

**Files:**
- 없음 (실행만)

**Step 1: ECoT 학습 시작**

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    adapter_train.py \
    --model ecot-7b \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/ecot-7b/v2-full \
    2>&1 | tee outputs/experiment_results/ecot-7b/v2-full/train.log
```

---

### Task 12: SpatialVLA V2-Full 어댑터 학습

**Files:**
- 없음 (실행만)

**Step 1: SpatialVLA 학습 시작**

```bash
CUDA_VISIBLE_DEVICES=5,6 accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    adapter_train.py \
    --model spatialvla-4b \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/spatialvla-4b/v2-full \
    2>&1 | tee outputs/experiment_results/spatialvla-4b/v2-full/train.log
```

---

### Task 13: TraceVLA V2-Full 어댑터 학습

**Files:**
- 없음 (실행만)

**Step 1: TraceVLA 학습 시작**

```bash
CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    adapter_train.py \
    --model tracevla-phi3v \
    --adapter_version 2 \
    --output_dir outputs/experiment_results/tracevla-phi3v/v2-full \
    2>&1 | tee outputs/experiment_results/tracevla-phi3v/v2-full/train.log
```

---

### Task 14: V2-Full 어댑터 평가 (4개 모델)

**Files:**
- 없음 (실행만, 각 모델 학습 완료 후)

**Step 1: OpenVLA V2-Full 평가**

```bash
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --model openvla-7b \
    --checkpoint outputs/experiment_results/openvla-7b/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/openvla-7b/v2-full/eval
```

**Step 2: ECoT V2-Full 평가**

```bash
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --model ecot-7b \
    --checkpoint outputs/experiment_results/ecot-7b/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/ecot-7b/v2-full/eval
```

**Step 3: SpatialVLA V2-Full 평가**

```bash
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --model spatialvla-4b \
    --checkpoint outputs/experiment_results/spatialvla-4b/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/spatialvla-4b/v2-full/eval
```

**Step 4: TraceVLA V2-Full 평가**

```bash
CUDA_VISIBLE_DEVICES=5 python adapter_eval.py \
    --model tracevla-phi3v \
    --checkpoint outputs/experiment_results/tracevla-phi3v/v2-full/checkpoints/best.pt \
    --output_dir outputs/experiment_results/tracevla-phi3v/v2-full/eval
```

---

### Task 15: Cross-Model 비교 리포트 생성

**Files:**
- Modify: `compare_adapter_results.py` (다중 모델 비교)

**Step 1: compare_adapter_results.py에 다중 모델 지원 추가**

```python
parser.add_argument("--models", nargs="+",
                    default=["openvla-7b", "ecot-7b", "spatialvla-4b", "tracevla-phi3v"],
                    help="Models to compare")
```

**Step 2: 결과 취합**

각 모델의 `eval_results.json`을 로드하여 하나의 비교표 생성:

```
| Model | Base MSE | V2-Full MSE | Improvement (%) |
|-------|---------|-------------|-----------------|
| OpenVLA-7B    | ? | ? | ? |
| ECoT-7B       | ? | ? | ? |
| SpatialVLA-4B | ? | ? | ? |
| TraceVLA      | ? | ? | ? |
```

**Step 3: 시각화**

- Per-model, per-dimension MSE bar chart
- Cross-model improvement comparison
- Sink type vs improvement correlation

**Step 4: Commit**

```bash
git add compare_adapter_results.py
git commit -m "feat(compare): cross-model adapter comparison report"
```

---

## 실행 순서 (의존성 기반)

```
Phase 0 — 코드 수정 (GPU 불필요, 순차적):
  [1] model_registry.py: action_type, source_layer, target_layers 추가
  [2] adapter_model.py: vision_tokens 파라미터화
  [3] extract_attention.py: load_model_from_registry 추가
  [4] attention_v3.py: 아키텍처별 패칭 일반화
  [5] adapter_train.py: multi-model 지원
  [5b] TraceVLA continuous loss 구현
  [6] adapter_eval.py: multi-model 평가 지원
  [7] run_adapter_experiment.py: base + v2-full only

Phase 1 — GPU 확보:
  [8] V1 학습 중단, GPU 해제

Phase 2 — OpenVLA 학습 (GPUs 1-4) + 동시에 Baseline 평가 (GPUs 5-7):
  [9] OpenVLA V2-Full 학습 (6-12h, GPUs 1-4)
  [10] 4개 모델 Baseline 평가 (병렬, GPUs 5-7)

Phase 3 — 나머지 모델 학습 (OpenVLA 완료 후 순차):
  [11] ECoT V2-Full 학습 (GPUs 1-4)
  [12] SpatialVLA V2-Full 학습 (GPUs 5-6)
  [13] TraceVLA V2-Full 학습 (GPUs 5-7)

Phase 4 — 평가 + 비교:
  [14] V2-Full 평가 (4개 모델)
  [15] Cross-model 비교 리포트

총 예상 시간: 코드 수정 2-3시간 + 학습/평가 24-40시간
```

---

## 검증 방법

1. **코드 수정 검증**: `python -c "from adapter_train import train"` 에러 없음
2. **OpenVLA backward compat**: `--model openvla-7b`로 기존과 동일하게 작동
3. **V2-Full 학습 loss 감소**: 각 모델에서 train loss가 단조 감소
4. **BlendAlpha 증가**: 0에서 서서히 증가 (학습 진행 확인)
5. **Test MSE 개선**: 4개 모델 모두 base 대비 V2-Full MSE 감소 (목표)
6. **Per-dimension 분석**: spatial/rotational 모두 개선 여부
7. **Cross-model 일관성**: 아키텍처에 무관하게 개선 관찰

---

## 출력 디렉토리 구조

```
outputs/experiment_results/
├── openvla-7b/
│   ├── base/eval/eval_results.json
│   └── v2-full/
│       ├── checkpoints/best.pt
│       ├── logs/training_log.json
│       └── eval/eval_results.json
├── ecot-7b/
│   ├── base/eval/eval_results.json
│   └── v2-full/
│       ├── checkpoints/best.pt
│       └── eval/eval_results.json
├── spatialvla-4b/
│   ├── base/eval/eval_results.json
│   └── v2-full/
│       ├── checkpoints/best.pt
│       └── eval/eval_results.json
├── tracevla-phi3v/
│   ├── base/eval/eval_results.json
│   └── v2-full/
│       ├── checkpoints/best.pt
│       └── eval/eval_results.json
└── cross_model_comparison.json
```
