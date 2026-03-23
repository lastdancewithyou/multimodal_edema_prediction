# Multi-Task Multimodal Learning for Pulmonary Edema Detection and Subtyping

## Overview

본 프로젝트는 중환자실(ICU) 환자의 폐부종(Pulmonary Edema) 발생을 조기에 예측하고, 폐부종의 원인을 심인성(Cardiogenic)과 비심인성(Non-cardiogenic)으로 분류하는 Multi-task 학습 모델입니다.

### 핵심 특징
- **Multi-task Learning**: 폐부종 발생 예측(Binary Classification)과 원인 분류(Subtype Classification)를 동시에 학습
- **Multimodal Fusion**: 시계열 생체신호, 흉부 X-ray 영상, 임상 텍스트, 인구통계학적 정보를 통합
- **Early Prediction**: 폐부종 발생 8시간 전 조기 예측 (기본값)
- **Temporal Modeling**: Segmented Cross-Attention 기반 시간적 패턴 학습

---

## Project Structure

```
clinical_multimodal_learning/
├── models/
│   ├── main_architecture.py       # Multi-task 메인 모델 (Encoder + Binary/Subtype Classifiers)
│   ├── encoder.py                 # Time-series Encoder (Transformer)
│   └── cxrformer_model.py         # CXFormer (Optional, 현재 DenseNet121 사용)
│
├── training/
│   ├── run.py                     # 학습 파라미터 설정
│   ├── data_processing.py         # Dataset, DataLoader, Sampler
│   ├── trainer.py                 # 학습 루프 (Single-stage)
│   ├── engine.py                  # Epoch 단위 학습/검증 로직
│   └── evaluator.py               # 모델 평가 및 메트릭 계산
│
├── loss/
│   ├── losses.py                  # Multi-task Loss Functions (BCE, CE, InfoNCE, SupCon 등)
│   └── target_generation.py       # Contrastive Learning용 Target 생성
│
├── analysis/
│   ├── umap_multitask.py          # UMAP 시각화
│   ├── calibration.py             # 모델 calibration 분석
│   └── decision_curve_analysis.py # Decision Curve Analysis
│
├── preprocess/                    # 전처리 Jupyter Notebooks
│   └── time_series_cxr_preprocess_20260316.ipynb
│
├── utils.py                       # 유틸리티 함수
├── main_train.py                  # 학습 실행 메인 파일
└── run.sh                         # 학습 실행 스크립트 (Accelerate)
```

---

## Model Architecture

### 1. Modality-Specific Encoders

#### Time-Series Encoder
- **Architecture**: Transformer-based Encoder (2 layers, 512-dim hidden)
- **Input**: 29개 생체신호 특징 (변수 값 + observed_mask)
- **Output**: [B, W, T, 512] - 시간 단계별 임베딩

#### Image Encoder
- **Model**: DenseNet121 (pretrained on MIMIC-CXR via torchxrayvision)
- **Freezing Strategy**: DenseBlock4와 최종 LayerNorm만 학습 가능
- **Output**: [B, W, T, 1024] - 영상 특징 임베딩

#### Text Encoder
- **Model**: BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- **Freezing Strategy**: Layer 10-11과 Pooler만 학습 가능
- **Output**: [B, W, T, 768] - [CLS] 토큰 임베딩

#### Demographic Encoder
- **Architecture**: MLP (2 layers)
- **Input**: 인구통계학적 변수 (나이, 성별 등)
- **Output**: [B, 256] - 환자 정보 임베딩

### 2. TS-Centric Cross-Attention Fusion

시계열 데이터를 중심으로 영상/텍스트 정보를 통합하는 Fusion 모듈입니다.

#### Core Components
```
1. Latent Array Initialization
   - L개의 learnable latent queries (기본값: 6개)
   - 각 latent는 시간 구간(segment)을 담당

2. Segmented Cross-Attention (Ablation 대상)
   - T개 time step을 L개 segment로 분할
   - 각 latent는 자신의 segment만 attention (local focus)
   - 대안: Full Global Attention (모든 latent가 모든 time step 관찰)

3. CLS Global Context (Optional)
   - CLS token이 과거 정보만 보는 causal attention
   - 각 latent에 global context 추가

4. Hierarchical Temporal Fusion
   - Latent 간 Self-Attention으로 시간적 구조 학습
```

#### Forward Pass Flow
```
Input: TS [B,W,T,512], IMG [B,W,T,1024], TXT [B,W,T,768]
   ↓
Cross-Attention (TS → Latent)  [L latents attend to time segments]
   ↓
Cross-Attention (IMG → Latent) [Latent attends to sparse images]
   ↓
Cross-Attention (TXT → Latent) [Latent attends to sparse texts]
   ↓
Self-Attention (Latent ↔ Latent) [Global temporal structure]
   ↓
Output: Fused Latent [B, W, L, 256]
```

### 3. Attention Pooling
- L개의 latent를 학습 가능한 attention weight로 pooling
- Output: [B, W, 256] - Window-level 임베딩

### 4. Multi-Task Prediction Heads

#### Binary Classifier (Edema Detection)
```python
edema_logits = Linear(256 → 1)  # [B, W, 1]
Loss: BCE (with label smoothing)
```

#### Subtype Classifier (Cardiogenic vs Non-cardiogenic)
```python
subtype_logits = Linear(256 → 2)  # [B, W, 2]
Loss: CE (only for edema=1 samples)
```

---

## Dataset & Data Processing

### 1. Data Merging
세 가지 DataFrame을 `stay_id`와 `hour_slot` 기준으로 outer join:
```python
merged_df = ts_df.merge(cxr_df).merge(text_df)
```

### 2. Window Generation
- **Sliding Window**: `window_size=1` (현재는 마지막 시점만 사용)
- **Prediction Horizon**: 8시간 (미래 라벨 할당)
- **Stride**: 1시간
- **Ablation Option**: `use_last_point_only=True` (시간 window 대신 단일 시점)

### 3. Label Structure
```python
# Multi-task labels
edema_labels: [B, W]      # 0: No edema, 1: Has edema, -1: Unlabeled
subtype_labels: [B, W]    # 0: Non-cardio, 1: Cardio, -1: Unlabeled
```

### 4. Efficient Batching
배치 내 중복 이미지/텍스트를 제거하여 메모리와 연산량 절약:
```python
unique_images: [N_unique, C, H, W]
unique_texts: [M_unique, max_length]
img_index_tensor: [B, W, T] → unique_images의 인덱스
text_index_tensor: [B, W, T] → unique_texts의 인덱스
```

### 5. Stratified Sampling
- **Patient-level stratification**: Edema 라벨 비율 유지
- **Batch composition**: 각 배치에 여러 환자의 window 포함
- **DDP alignment**: drop_last=True로 GPU 간 배치 수 동기화

---

## Training Pipeline

### Single-Stage Training (현재 사용)

```python
Total Loss = w1 * BCE(edema) + w2 * CE(subtype) + w3 * InfoNCE + w4 * SupCon
```

#### Loss Components
1. **Binary Cross-Entropy (BCE)**: Edema detection
   - Label smoothing 적용 (기본값: 0.1)

2. **Cross-Entropy (CE)**: Subtype classification
   - Edema=1인 샘플에만 적용
   - Label smoothing 적용

3. **Temporal InfoNCE (Optional)**: Unsupervised contrastive learning
   - 시간적으로 가까운 window는 positive
   - 다른 환자의 window는 negative

4. **SupCon (Optional)**: Supervised contrastive learning
   - 같은 라벨의 window끼리 positive

#### Training Configuration
```python
Epochs: 30
Learning Rate: 1e-4
Optimizer: AdamW
Scheduler: ReduceLROnPlateau
Early Stopping: patience=5 (validation AUROC 기준)
```

### Two-Stage Training (Optional, 현재 미사용)

#### Stage 1: Contrastive Pretraining
```python
Loss = InfoNCE + SupCon
Epochs: 40
Learning Rate: 1e-4
```

#### Stage 2: Classification Fine-tuning
```python
Loss = BCE + CE
Epochs: 30
Learning Rate: 5e-5
Options: Linear Probing (encoder frozen) vs Fine-tuning
```

---

## Key Features & Techniques

### 1. Segmented Temporal Attention
시간축을 segment로 나눠 각 latent가 local region에 집중:
```python
# T=6 time steps, L=6 latents
Latent 0 → Time [0:1]
Latent 1 → Time [1:2]
...
Latent 5 → Time [5:6]
```
**장점**: 시간적 granularity 확보, 계산 효율성

### 2. Sparse Modality Handling
영상/텍스트는 일부 time step에만 존재:
```python
has_cxr: [B, W, T]    # 1 if CXR exists at this timestep
has_text: [B, W, T]   # 1 if text exists at this timestep
key_padding_mask: Mask out missing modalities in attention
```

### 3. Multi-Task Learning Strategy
```python
# Hierarchical task structure
Task 1 (Main): Edema Detection (Binary)
Task 2 (Sub): Subtype Classification (P(subtype|edema=1))

# Loss masking
CE loss는 edema=1인 샘플에만 적용
BCE loss는 모든 유효 라벨에 적용
```

### 4. Label Smoothing
Hard label 대신 soft label 사용으로 과적합 방지:
```python
# Binary (epsilon=0.1)
0 → 0.05, 1 → 0.95

# Multi-class (epsilon=0.1)
[1,0,0] → [0.9, 0.05, 0.05]
```

---

## Evaluation Metrics

### Binary Task (Edema Detection)
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Accuracy, Precision, Recall, F1-score
- Specificity, NPV (Negative Predictive Value)

### Subtype Classification
- AUROC (Cardiogenic vs Non-cardiogenic)
- AUPRC
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix

### Additional Analysis
- Calibration Curve
- Decision Curve Analysis
- UMAP Visualization

---

## Usage

### 1. Environment Setup
```bash
# Install dependencies
pip install torch torchvision
pip install transformers accelerate
pip install torchxrayvision
pip install scikit-learn pandas numpy
pip install wandb umap-learn matplotlib seaborn
```

### 2. Data Preparation
전처리된 데이터 경로 설정:
```python
# In data_processing.py
CACHED_IMAGE_DIR = "/path/to/cached_images_224_0317/"
```

데이터 형식:
- `ts_df`: [stay_id, hour_slot, features...]
- `cxr_df`: [stay_id, hour_slot, hash_path, cxr_flag]
- `text_df`: [stay_id, hour_slot, tokenized_text, text_flag]
- `demo_df`: [hadm_id, demographic_features...]

### 3. Training
```bash
# Single GPU (GPU 0)
bash run.sh

# Or directly with accelerate
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file config_single_gpu_0.yaml \
  main_train.py
```

### 4. Configuration
[training/run.py](training/run.py:1)에서 하이퍼파라미터 설정:

```python
# Core settings
experiment_id = "36"
train_batch_size = 64
single_stage_epochs = 30
single_learning_rate = 1e-4

# Multi-task weights
bce_weight = 1.0    # Edema detection
ce_weight = 1.0     # Subtype classification
ucl_weight = 0.0    # Temporal InfoNCE (disabled)
scl_weight = 0.0    # SupCon (disabled)

# Ablation flags
use_segmented_attention = True   # Segmented vs Full Global Attention
use_cls_global = False           # CLS token for causal context
use_last_point_only = True       # Single timepoint vs window
disable_cxr = False              # Turn off image modality
disable_txt = False              # Turn off text modality
use_demographic = True           # Use demographic features
```

### 5. Model Checkpoints
```python
# Best model saved to:
./output/checkpoints/experiment_{id}/best_model.pth

# Stage 1 model (if two-stage):
./output/stage1_models/experiment_{id}_stage1.pth
```

---

## Ablation Studies

프로젝트에서 수행 가능한 Ablation 실험:

### 1. Fusion Architecture
```python
use_segmented_attention = True/False
# True: Segmented local attention
# False: Full global attention
```

### 2. Temporal Context
```python
use_cls_global = True/False
# True: Add causal global context via CLS token
# False: No global context
```

### 3. Temporal Window
```python
use_last_point_only = True/False
window_size = 1/6/12
# Test single timepoint vs sliding window
```

### 4. Modality Contribution
```python
disable_cxr = True/False  # Image ablation
disable_txt = True/False  # Text ablation
use_demographic = True/False  # Demographic ablation
```

### 5. Loss Functions
```python
use_bce = True/False         # Binary edema loss
use_ce = True/False          # Subtype classification loss
use_temporal_ucl = True/False  # Temporal contrastive
use_supcon = True/False      # Supervised contrastive
```

---

## Implementation Details

### Memory Optimization
1. **Gradient Checkpointing**: CXFormer 사용 시 활성화
2. **Unique Batching**: 배치 내 중복 이미지/텍스트 1회만 인코딩
3. **Mixed Precision**: Accelerate 자동 지원
4. **Valid Window Masking**: 패딩된 window는 연산 제외

### Computational Efficiency
```python
# Image caching
self.image_cache: Dict[hash_path, tensor]  # 자주 쓰이는 이미지 캐싱

# Sparse modality processing
if unique_images.numel() > 0:
    features = img_encoder(unique_images)
# → 존재하는 이미지만 인코딩
```

### Reproducibility
```python
random_seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
seed_worker()  # DataLoader worker seed 고정
```

---

## Future Directions

### Potential Improvements
1. **Time2Vec Integration**: 절대 시간 정보 인코딩 (현재 주석 처리)
2. **CXFormer Migration**: DenseNet → CXFormer 전환
3. **Dynamic Segment Allocation**: 고정 segment 대신 학습 가능한 분할
4. **Attention Visualization**: Segmented attention pattern 시각화
5. **External Validation**: 다른 병원 데이터셋 검증

### Research Questions
- Segment 개수(L)의 최적값은?
- Iterative fusion 횟수(num_iterations)의 영향은?
- 어떤 modality가 각 task에 가장 중요한가?
- Early prediction horizon을 더 늘릴 수 있는가?

---