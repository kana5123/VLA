# Attention Enhancement Experiment Requirements

## Original Requirement

> OpenVLA로 10개 에피소드 추론 시 attention이 텍스트에 치중되어 있으므로,
> 텍스트 지시문이 가리키는 객체에 해당하는 이미지 패치의 attention score를 강화하여
> 기존 baseline MSE와 비교하고 싶다.
> "Attention Speaks Volumes" 등의 방법론을 참고하여 다양한 강화 알고리즘을 적용하고,
> 타임스텝별 MSE를 비교하겠다.

## Clarified Requirements

### Goal

OpenVLA 추론 시 **객체 관련 이미지 패치의 attention을 강화**하여
action prediction MSE가 개선되는지 실험

### Pipeline Overview

```
1. Baseline 측정
   10 에피소드 x 각 타임스텝 -> OpenVLA 추론 -> predicted action vs GT action -> MSE

2. Object Grounding
   지시문에서 명사 자동 추출 (spacy) -> GroundingDINO로 객체 검출 ->
   바운딩 박스 -> 해당 16x16 패치 인덱스 매핑

3. Attention 강화 (추론 시점, 3가지 방법)
   (1) Logit Bias: softmax 전 attention logits에 객체 패치 위치 가산
   (2) Weight Rescale: softmax 후 attention weights 재분배 (Atlas 방식)
   (3) Head Steering: 특정 head만 선택적 강화

4. Enhanced 측정
   각 강화 방법별 -> predicted action vs GT action -> MSE

5. 비교
   - 에피소드 평균 MSE (baseline vs 강화 방법 3가지)
   - 타임스텝별 7차원 전체 MSE 그래프
```

### Decisions

| Question | Decision |
|----------|----------|
| Object Grounding method | GroundingDINO + spacy noun extraction |
| Intervention timing | Inference-time only (no fine-tuning) |
| MSE granularity | Episode average + per-timestep 7-dim aggregate MSE |
| Enhancement algorithms | 3 methods: logit bias, weight rescale, head steering |
| Object query source | Auto-extracted nouns from instruction text |

### Technical Details

#### 1. Baseline MSE

- Run OpenVLA inference on 10 episodes (all timesteps)
- De-tokenize predicted action token IDs to continuous values
- Compute MSE: `mean((pred_action - gt_action)^2)` per timestep
- Aggregate per-episode average

#### 2. Object Grounding Pipeline

- Extract nouns from instruction using spacy (`en_core_web_sm`)
  - Example: "put small spoon from basket to tray" -> ["spoon", "basket", "tray"]
- Run GroundingDINO on each step's image with extracted nouns as queries
- Convert bounding boxes to 16x16 patch grid indices
  - `patch_row = int(bbox_center_y / image_height * 16)`
  - `patch_col = int(bbox_center_x / image_width * 16)`
- Store object patch indices per step

#### 3. Attention Enhancement Methods

**Method 1: Logit Bias (pre-softmax)**
- Hook into attention computation before softmax
- Add bias term `alpha` to attention logits at object patch positions
- `logits[..., object_patch_indices] += alpha`
- Hyperparameter: alpha (search range)

**Method 2: Weight Rescale (post-softmax, Atlas-inspired)**
- After softmax, scale attention weights for object patches
- `weights[..., object_patch_indices] *= lambda_scale`
- Re-normalize to maintain valid probability distribution
- Reference: Atlas (Attention Speaks Volumes, ACL 2025)
  - Layer-wise greedy search for optimal lambda
  - Target top-k layers with highest object attention delta

**Method 3: Head Steering**
- Identify which attention heads naturally attend to vision tokens
- Amplify those heads' contributions to object patches
- Leave other heads unchanged
- Prevents catastrophic distribution shift

#### 4. Outlier Prevention

- All methods must maintain valid attention distributions (sum to 1)
- Weight rescale uses re-normalization after scaling
- Logit bias uses bounded alpha values
- Head steering only modifies subset of heads

### Metrics

- **Primary**: MSE per timestep (7-dim aggregate)
- **Aggregate**: Episode-average MSE
- **Comparison**: Baseline vs 3 enhancement methods

### References

- Attention Speaks Volumes (Atlas): https://arxiv.org/abs/2410.22517
  - Key: layer-wise attention scaling with lambda in [0,1]
  - Greedy search over top-k=3 layers
- Qwen-LookAgain: https://arxiv.org/html/2505.23558v2
  - Visual token re-attention in VLMs

### Dependencies (additional)

- `spacy` + `en_core_web_sm` model (noun extraction)
- `groundingdino` (object detection)
- Already have: torch, transformers, PIL, numpy, matplotlib

### Data

- 10 episodes from BridgeData V2 (already downloaded)
- 385 total timesteps
- Each step has: image, instruction, 7-dim ground truth action
