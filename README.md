# 🛒 온라인 채널 제품 판매량 예측 (PyTorch LSTM)

> **모델 전략:** 전체 제품을 하나의 글로벌 LSTM으로 학습 (Global Time-Series Model)  
> **입력:** 과거 28일 시퀀스 → **출력:** 미래 21일 판매량 (Direct Multi-Step)  
> **평가지표:** MAE (Mean Absolute Error)

---

## 📌 프로젝트 목적

과거의 제품 판매량, 브랜드 검색량(Keyword) 등의 시계열 데이터를 활용하여  
향후 21일간의 제품별 판매량을 예측하는 딥러닝 모델 구축

> **예측 대상:** 약 15,890개 제품의 **미래 21일** 일별 판매 수량  
> **학습 데이터:** 2022-01-01 ~ 2023-04-04 (459일)  
> **예측 기간:** 2023-04-05 ~ 2023-04-25 (21일)

---

## 📁 프로젝트 구조

```
Sales/
├── README.md
├── Sales_Merged_v1.ipynb       # 전체 파이프라인 노트북 (현재 파일)
├── korean_font.py              # 한글 폰트 설정 모듈
├── data/
│   ├── train.csv               # 학습 데이터 (일별 판매 수량)
│   ├── sales.csv               # 학습 데이터 (일별 매출액)
│   ├── product_info.csv        # 제품 메타 정보
│   ├── brand_keyword_cnt.csv   # 브랜드 키워드 검색량
│   └── sample_submission.csv   # 제출 양식
└── output/
    └── submission.csv          # 최종 예측 결과 (생성 예정)
```

---

## 📊 데이터 설명

| 파일명 | 설명 |
|---|---|
| `train.csv` | 제품별 일별 **판매 수량** (2022-01-01 ~ 2023-04-04, 약 15,890 제품) |
| `sales.csv` | 제품별 일별 **매출액** (동일 기간, train과 동일 구조) |
| `product_info.csv` | 제품 특성 정보 (제품유형, 용량, 피부타입, 섭취방법 등 텍스트 피처) |
| `brand_keyword_cnt.csv` | 브랜드별 일별 키워드 검색량 (정규화 값) |
| `sample_submission.csv` | 예측 대상 기간 **2023-04-05 ~ 2023-04-25** (21일) |

### 컬럼 구조 (train.csv / sales.csv)

| 컬럼 | 설명 |
|---|---|
| `ID` | 제품 고유 번호 |
| `제품` | 제품 코드 |
| `대분류` / `중분류` / `소분류` | 카테고리 계층 |
| `브랜드` | 브랜드 코드 |
| `2022-01-01` ~ `2023-04-04` | 일별 판매 수량 / 매출액 |

---

## 🔄 파이프라인 개요

```
STEP 1  환경 설정 및 데이터 로드
   ↓
STEP 2  탐색적 데이터 분석 (EDA) + 시각화  ✅ 완료
        (2-1 카테고리 구조 ~ 2-13 공휴일 분석)
   ↓
STEP 3  데이터 전처리  ✅ 완료
        (급락 구간 처리 / 클리핑 / Long 변환 / 마스킹 / 인코딩 / 피처 병합)
   ↓
STEP 4  피처 엔지니어링  🔜 예정
        (Lag, Rolling, Date Features)
   ↓
STEP 5  모델링  🔜 예정
        (Baseline → LSTM → 앙상블 비교)
   ↓
STEP 6  모델 성능 보고 및 최종 선택
   ↓
STEP 7  최종 예측 및 제출 파일 생성
```

---

## 🔍 EDA 분석 목록 (STEP 2)

| # | 분석 항목 | 전처리 활용 |
|---|----------|-----------|
| 2-1 | 카테고리 구조 + 제품 수 / 판매량 분포 | 임베딩 레이어 설계 |
| 2-2 | 판매량 분포 — Long Tail + 로렌츠 곡선 | log1p 변환 적용 근거 |
| 2-3 | 일별 판매 트렌드 — 급락 구간 탐지 | 2023-02-20 이후 처리 방법 결정 |
| 2-4 | 요일·월별 판매 패턴 | `day_of_week`, `month`, `quarter` 피처 |
| 2-5 | Spike(이상치) 탐지 — MA30 ± 2σ | 브랜드별 클리핑 처리 |
| 2-6 | 구조적 0 vs 진짜 0 구분 | 출시 전 / 단종 후 마스킹 |
| 2-7 | 브랜드 검색량 — 판매 예측 기여도 | `kw_log`, `kw_lag_1`, `kw_lag_3` 채택 |
| 2-8 | ACF / PACF — 주간 주기성 확인 | `lag_1`, `lag_7` 피처 설계 |
| 2-9 | CCF — 검색량 선행 효과 (Lag 2 피크) | `kw_lag_1`, `kw_lag_3` 설계 |
| 2-10 | 시계열 분해 (STL) — 추세·계절·잔차 | 급락 구간 확인 |
| 2-11 | ADF 정상성 검정 — 차분 필요 여부 | log1p 변환으로 안정화 |
| 2-12 | 대분류별 시즌성 히트맵 | 카테고리 임베딩 레이어 강화 |
| 2-13 | 한국 공휴일 vs 판매량 | `is_holiday`, `days_to_holiday` 피처 |

---

## ⚙️ 데이터 전처리 상세 (STEP 3)

### 3-0. 급락 구간 처리 (5가지 방법 비교)

| 방법 | 전략 | 선택 여부 |
|------|------|---------|
| Method 1 | 급락 기간 날짜 컬럼 제외 | — |
| Method 2 | 전년 동기 × 성장률 합성 복원 | — |
| Method 3 | 이동평균 7일 + 요일 보정 교체 | — |
| Method 4 | 이동평균 30일 + 요일 보정 교체 | — |
| **Method 5** | **원본 유지 + `is_post_crash` 플래그 생성** | ✅ **선택** |

> **Method 5 선택 이유:** 가정이 가장 적음. Method 2~4는 존재하지 않는 가짜 값을 학습 데이터에 삽입하는 반면, Method 5는 원본을 유지하고 급락 여부를 플래그로 제공해 모델이 직접 판단하게 함.

### 3-1 ~ 3-7 전처리 단계

| 단계 | 내용 |
|------|------|
| **3-1** | Spike 클리핑 — 개별 제품 이상치 (99% quantile 기준) |
| **3-2** | 단가(Price) 파생 — `매출액 ÷ 판매량`, qty=0인 날은 ffill |
| **3-3** | Wide → Long 포맷 변환 (약 730만 행) |
| **3-4** | 구조적 0 마스킹 — 첫 판매일 이전 / 마지막 판매 +90일 이후 |
| **3-5** | 카테고리 인코딩 — `LabelEncoder` (대/중/소분류, 브랜드, 제품) |
| **3-6** | 브랜드 검색량 병합 — `kw_log`, `kw_lag_1`, `kw_lag_3` |
| **3-7** | Product Info 특성 파싱 — ⏸️ **현재 비활성화** (`USE_PRODUCT_INFO = False`) |

> **3-7 비활성화 사유:** 매칭률 65.9% (5,398개 제품 정보 없음), 텍스트 비정형.  
> Baseline 성능 확인 후 필요 시 `USE_PRODUCT_INFO = True`로 활성화.

---

## 📋 현재 피처 목록 (STEP 3 완료 기준)

| 그룹 | 피처 |
|------|------|
| 식별자 | `ID`, `date` |
| 타겟 | `qty` |
| 카테고리 | `대분류_enc`, `중분류_enc`, `소분류_enc`, `브랜드_enc`, `product_enc` |
| 가격 | `price` |
| 검색량 | `kw_log`, `kw_lag_1`, `kw_lag_3` |
| 마스크/플래그 | `is_structural_zero`, `is_post_crash` |

> **STEP 4에서 추가 예정:** `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_7`, `rolling_14`, `rolling_28`, `day_of_week`, `month`, `quarter`, `is_holiday`, `days_to_holiday`, `is_weekend`

---

## 🏗️ 모델 아키텍처 (예정)

```
Input (sequence_length=28 × n_features)
        ↓
   LSTM Layer 1  (hidden_size=128, dropout)
        ↓
   LSTM Layer 2  (hidden_size=64, dropout)
        ↓
   Fully Connected (64 → 32 → 21)
        ↓
   Output: 예측 판매량 (미래 21일)
```

---

## 🎯 학습 전략 (예정)

| 항목 | 설정 |
|------|------|
| Train / Validation | 시간 기준 분리 (마지막 N일을 Validation) |
| Loss Function | MAE (Mean Absolute Error) |
| Optimizer | Adam (lr=1e-3) |
| EarlyStopping | val_loss 기준, patience=10 |
| ReduceLROnPlateau | val_loss 개선 없을 시 lr 감소 |

---

## 🚀 실행 방법

```bash
# 1. 환경 설정
pip install torch pandas numpy matplotlib seaborn scikit-learn statsmodels holidays tqdm

# 2. 한글 폰트 모듈 확인 (같은 폴더에 korean_font.py 필요)

# 3. 경로 설정 (노트북 내)
# HOME = os.getcwd()  # 노트북이 있는 디렉토리

# 4. 노트북 실행
jupyter notebook Sales_Merged_v1.ipynb
```

---

## 📦 필요 라이브러리

```
torch >= 2.0
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
matplotlib >= 3.6
seaborn >= 0.12
statsmodels
holidays
tqdm
```

> `holidays`, `statsmodels`는 노트북 STEP 1 실행 시 자동 설치됩니다.

---

## 📈 예측 대상

- **기간:** 2023-04-05 ~ 2023-04-25 (21일)
- **단위:** 제품별 일별 판매 수량
- **제출 형식:** `sample_submission.csv` 형식에 맞춰 예측값 기입
