# 🛒 온라인 채널 제품 판매량 예측 (PyTorch LSTM)

> **주제:** 온라인 채널 제품 일별 판매량 시계열 예측  
> **모델:** PyTorch LSTM (딥러닝 기반 시계열 모델)  
> **평가지표:** MAE (Mean Absolute Error)

---

## 📁 프로젝트 구조

```
Sales
├── README.md
├── Sales.ipynb                 # 전체 파이프라인 노트북
├── data/
│   ├── train.csv               # 학습 데이터 (일별 판매 수량)
│   ├── sales.csv               # 학습 데이터 (일별 매출액)
│   ├── product_info.csv        # 제품 메타 정보
│   ├── brand_keyword_cnt.csv   # 브랜드 키워드 검색량
│   └── sample_submission.csv   # 제출 양식
└── output/
    └── submission.csv          # 최종 예측 결과
```

---

## 📊 데이터 설명

| 파일명 | 설명 |
|---|---|
| `train.csv` | 제품별 일별 **판매 수량** (2022-01-01 ~ 2023-04-04, 약 15,890 제품) |
| `sales.csv` | 제품별 일별 **매출액** (동일 기간, train과 동일 구조) |
| `product_info.csv` | 제품 특성 정보 (제품유형, 용량, 섭취방법 등 텍스트 피처) |
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
STEP 0  라이브러리 불러오기 & 데이터 로드
   ↓
STEP 1  EDA — 데이터 탐색 및 시각화
   ↓
STEP 2  데이터 전처리 & 피처 엔지니어링
        (Lag, Rolling, Date Features, Keyword Features)
   ↓
STEP 3  딥러닝 모델 구성 (PyTorch LSTM)
   ↓
STEP 4  컴파일 (MAE Loss + Adam Optimizer)
   ↓
STEP 5  Callbacks 정의 (EarlyStopping, ReduceLROnPlateau)
   ↓
STEP 6  모델 학습 및 검증
   ↓
STEP 7  학습 결과 및 예측 시각화
   ↓
STEP 8  최종 예측 및 제출 파일 생성
```

---

## 🏗️ 모델 아키텍처

```
Input (sequence_length × n_features)
        ↓
   LSTM Layer 1  (hidden_size=128, dropout)
        ↓
   LSTM Layer 2  (hidden_size=64, dropout)
        ↓
   Fully Connected (64 → 32 → 1)
        ↓
   Output: 예측 판매량 (다음 21일)
```

---

## ⚙️ 주요 피처 엔지니어링

| 피처 유형 | 설명 |
|---|---|
| **Lag Features** | 1일, 7일, 14일, 28일 전 판매량 |
| **Rolling Features** | 7일, 14일, 28일 이동 평균 / 이동 표준편차 |
| **Date Features** | 요일, 월, 주차, 공휴일 여부, 주말 여부 |
| **Keyword Features** | 브랜드별 키워드 검색량 lag / rolling |
| **Category Features** | 대/중/소분류, 브랜드 Label Encoding |
| **Sales Features** | 매출액 기반 Lag / Rolling (price proxy) |

---

## 🎯 학습 전략

- **Train / Validation split:** 시간 기준 분리 (마지막 N일을 Validation)
- **Loss Function:** MAE (Mean Absolute Error)
- **Optimizer:** Adam (lr=1e-3)
- **EarlyStopping:** val_loss 기준, patience=10
- **ReduceLROnPlateau:** val_loss 개선 없을 시 lr 감소

---

## 🚀 실행 방법

```bash
# 1. 환경 설정
pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm

# 2. 경로 설정 (노트북 내)
import os
HOME = os.getcwd()  # '/mnt/c/users/min2m/github/Sales/권희민'

# 3. 노트북 실행
jupyter notebook main.ipynb
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
tqdm
```

---

## 📈 예측 대상

- **기간:** 2023-04-05 ~ 2023-04-25 (21일)
- **단위:** 제품별 일별 판매 수량
- **제출 형식:** `sample_submission.csv` 형식에 맞춰 예측값 기입
