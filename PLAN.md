# Plan: v7 강화 + BERT 통합 로드맵

## Context

현재 fusion은 v7 (TS slot이 query carrier, CXR/Text가 KV)로 라우팅되어 있음. 사용자는 v7가 후속 작업(BERT 토큰 시퀀스 → slot-level refinement → distillation)의 paradigm 기반이 된다는 점을 확인하고, **TS 슬롯이 cross-attention 전에 시간 컨텍스트를 한 번 정리**할 필요를 인식. 현재 v7는 raw TS에 self-attention이 없어서 TS 슬롯이 자기 컨텍스트 없이 곧바로 CXR/Text와 fusion됨 → TS query의 표현력이 부족할 수 있음.

이번 plan은:
1. **Step 1 (즉시)**: v7에 TS self-attention block 추가 — 작은 코드 수정
2. **Step 2 (v7 검증 후)**: BERT 통합 + 토큰 시퀀스 text → v7 text_cross_attn KV 변경
3. **Step 3 (장기)**: Slot-level distillation (단순 preview, 별도 plan)

---
## Step 1은 완료됨. 

## Step 2: BERT 통합 + 토큰 시퀀스 Text → v7 KV

### 전제

사용자가 전처리 노트북에서 텍스트 임베딩 대신 `{'input_ids': [128], 'attention_mask': [128]}`를 .pt로 전처리 완료함.. 학습 시 모델이 직접 BERT forward.

### 변경 1: `training/data_processing.py`

기존 `load_text_emb`를 토큰 dict 반환으로 변경. collate에서 stack:

```python
def load_text_emb(self, path):
    d = self._load_emb(path, self.text_emb_cache)
    return d   # {'input_ids': [128], 'attention_mask': [128]}

# collate 안 (line 303~352 영역):
if unique_text_paths:
    loaded = [self.load_text_emb(p) for p in unique_text_paths]
    unique_text_input_ids = torch.stack([x['input_ids'].long() for x in loaded])    # [N, 128]
    unique_text_attn_mask = torch.stack([x['attention_mask'].long() for x in loaded])  # [N, 128]
else:
    unique_text_input_ids = torch.empty(0, 128, dtype=torch.long)
    unique_text_attn_mask = torch.empty(0, 128, dtype=torch.long)

batch['unique_text_input_ids'] = unique_text_input_ids
batch['unique_text_attn_mask'] = unique_text_attn_mask
# 기존 'unique_text_embs' 제거
```

### 변경 2: `training/engine.py`

`prepare_multiview_inputs`의 text_data dict 구조 갱신:

```python
text_data = {
    'unique_input_ids': batch['unique_text_input_ids'],
    'unique_attn_mask': batch['unique_text_attn_mask'],
    'unique_indices':   unique_indices,
    'positions':        valid_positions,
}
```

### 변경 3: `models/main_architecture.py`

**`MultiModalEncoder.__init__`**:

```python
from transformers import AutoModel
if not disable_txt:
    self.text_encoder = AutoModel.from_pretrained(args.text_model_path)
    for p in self.text_encoder.parameters():
        p.requires_grad = False
    self.text_encoder.eval()
```

**`MultiModalEncoder.forward` 라인 192–213 Text 섹션 교체**:

```python
if not self.disable_txt:
    with timer("Text BERT forward", None):
        uniq_ids  = text_data['unique_input_ids'].to(device)
        uniq_mask = text_data['unique_attn_mask'].to(device)

        if uniq_ids.numel() > 0:
            with torch.no_grad():
                bert_out = self.text_encoder(
                    input_ids=uniq_ids, attention_mask=uniq_mask,
                )
            # [N_uniq, 128, 768] — autocast로 fp16/bf16 자동 캐스트
            unique_text_tokens = bert_out.last_hidden_state.to(dtype=ts_embeddings.dtype)
        else:
            unique_text_tokens = torch.empty(0, 128, self.text_emb_dim,
                                             device=device, dtype=ts_embeddings.dtype)

    # 윈도우별 텍스트 KV bank 구성: window 내 unique text events들의 토큰을 concat
    text_kv_bank, text_kv_mask = self._build_text_kv_bank(
        unique_text_tokens, uniq_mask,
        text_data['unique_indices'], text_data['positions'],
        B=B, max_tokens=128,
    )
else:
    text_kv_bank = torch.empty(B, 0, self.text_emb_dim,
                               device=device, dtype=ts_embeddings.dtype)
    text_kv_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
```

**신규 헬퍼 `_build_text_kv_bank`** (`MultiModalEncoder` 내):
- `positions [num_pos, 2]`로부터 각 batch row에 속한 text event들 식별
- 그 event들의 valid 토큰만 concat
- 모든 batch row의 bank를 같은 길이(max_in_batch)로 padding
- 반환: `text_kv_bank [B, L_max, 768]`, `text_kv_mask [B, L_max]` (1=valid)

**`text_pre_proj` 입력**: `text_kv_bank` → `[B, L_max, align_d_model]`로 사영.

### 변경 4: v7 text_cross_attn 입력 갱신

`MultiModalEncoder.forward` 라인 247 근처에서 v7에 전달하는 인자:

```python
fused_kwargs = dict(
    ts_embeddings = ts_emb_p,
    img_embeddings = img_emb_p,
    text_embeddings = text_kv_bank_p,         # ← 윈도우별 token bank, [B, L_max, d]
    text_key_padding_mask = ~text_kv_mask,    # ← bank용 mask
    time_indices = time_idx,
    img_time_indices = img_time_idx,
    img_key_padding_mask = ~has_img_flat,
)
```

v7 코드 자체는 text_embeddings를 KV로 받는 인터페이스 그대로라서 **수정 거의 없음**. 단:
- `time2vec_txt`는 토큰 단위 시간을 정의하기 어려우므로 비활성화 (윈도우당 text는 시간 분해능이 토큰 수준이 아님). 라인 503, 507, 534, 538의 time2vec_txt/text_with_time을 단순화:
  ```python
  text_with_time = text_embeddings   # time2vec_txt 적용 X
  ```
  (또는 윈도우 시작 시점 1개를 broadcast해서 더하기 — 차후 결정)

### Alignment anchor 처리

기존 `_build_alignment_anchors`는 `text_emb_p [B, seq_len, d]`을 받아 masked mean으로 `text_pool [B, d]` 생성. 새 구조에서는:

```python
# text_kv_bank_p [B, L_max, d] + text_kv_mask [B, L_max]
text_pool, has_text_any = self._masked_mean(text_kv_bank_p, ~text_kv_mask)
```

`_masked_mean`은 그대로 사용. `losses.alignment_loss`는 변경 없음 (입력 shape [B, d] 유지).

### `training/run.py` 추가 args

```python
parser.add_argument('--text_model_path', type=str,
    default='/home/DAHS1/gangmin/my_research/clinical_multimodal_learning/models/bio_clinicalbert_safetensors',
    help='Bio_ClinicalBERT pretrained path (모델 내 frozen forward)')
```

`--text_emb_dim`은 768 그대로.

### Step 2 검증

1. `text_model_path` 디렉토리 존재 확인 (`config.json`, `model.safetensors`, `tokenizer.json` 확인 완료됨)
2. import & 모델 초기화 smoke test
3. 1 epoch 학습 — BERT forward NaN 없음, GPU OOM 없음, step time 점검
   - 예상 step time 증가: 10~30ms/batch (BERT-base on ~100 unique events)
4. v7 (Step 1) baseline 대비 val AUROC 비교
   - 동등 이상이면 토큰 시퀀스 표현이 최소 정보 손실 없이 통합된 것

### Step 2 위험요소

- **GPU 메모리**: BERT-base 220MB + batch당 BERT activations. batch=512 그대로 가능한지 첫 step에서 확인. 안 되면 batch 384/256로 낮춤.
- **BERT eval mode 유지**: `model.train()` 호출이 BERT에도 적용되지 않게 `self.text_encoder.eval()` 강제. `MultiModalEncoder` train mode override 필요할 수 있음.
- **Autocast 호환**: BERT가 fp16 autocast에서 안정적인지 첫 epoch 모니터링.

---

## Step 3 Preview (별도 plan)

v7 + BERT 통합이 성공하면 Phase 3로 slot-level distillation:

- **New module**: `TextAnchoredSlotRefiner` — TS 슬롯(Q) × text token bank(KV) → text-augmented TS 슬롯
- v7의 text_cross_attn은 latent 없으니 사실상 이 역할에 가깝지만, distillation을 위해 **명시적으로 "with-text path"와 "no-text path"를 single forward 안에 분리** 필요
- Loss: `L_distil = MSE(ts_query_no_text, sg(ts_query_with_text))` — text 있는 윈도우에서만
- BCE는 with-text path로 학습 (teacher signal)
- 추론은 no-text path 사용 — 학습 중 distill된 weights가 text 지식을 흡수한 상태

이 단계는 본 plan 범위 밖, v7+BERT 검증 후 별도 plan으로 작성.

---

## Files Summary

| 파일 | Step 1 | Step 2 |
|---|---|---|
| `models/main_architecture.py` | v7에 TS self-attn block 추가 | BERT 로드, forward Text 섹션 교체, `_build_text_kv_bank` 신규, v7 text 인자 변경 |
| `training/data_processing.py` | — | text load → input_ids/attn_mask dict, collate에서 stack |
| `training/engine.py` | — | text_data dict 키 변경 |
| `training/run.py` | — | `--text_model_path` arg 추가 |
| `loss/losses.py` | — | 변경 없음 |

## End-to-end Verification

Step 1 + Step 2 모두 적용 후:
1. `bash run.sh` 또는 `python main_train.py` 1~2 epoch
2. wandb 확인: train BCE 감소, val AUROC w/o text 정상 기록, alignment loss 정상
3. GPU 메모리/step time 점검
4. 최종 비교 표:
   | Run | 구성 | val AUROC w/o text |
   |---|---|---|
   | 직전 v7 (self-attn 없음) | baseline | 0.74 가정 |
   | v7 + TS self-attn | Step 1 | ? |
   | v7 + TS self-attn + BERT 토큰 KV | Step 1+2 | ? |
5. Step 2가 +0.02 이상 개선되면 Phase 3 (distillation) plan 진행
