# brain_test — EEG 집중도 모니터링 & V-패턴 감지

실시간 뇌파(EEG)로 집중도를 시각화하고, 집중도가 급락한 뒤 빠르게 회복하는
**V-패턴**을 감지하는 시스템입니다. 3D 두피 토포그래피, 주파수 대역 분석,
좌우뇌 균형, 실험 프로토콜, 개인 보정을 지원합니다.

---

## 주요 기능

- 🧠 **3D 뇌 시각화** (three.js) — 두피 토포그래피 · 전극 활동 · 히트맵
- 📊 **집중도 지수** — Engagement Index `β / (α + θ)` + 주파수 대역 `δ/θ/α/β/γ`
- ⚡ **V-패턴 감지** — 룰베이스 + EEGNet+LSTM 머신러닝 모델
- 🎛️ **다중 데이터 소스** — 시뮬레이션 · 공개 데이터셋 · 실제 뇌파 기기
- 🎯 **개인 보정 + 실험 프로토콜** (baseline / relax / task)

---

## 빠른 시작

### 방법 1 — 라이브 서버 (실시간)

```bash
pip install websockets numpy scipy pandas
python eeg_server.py --source sim
```

그다음 브라우저에서 **`mockup_3d.html`** 을 열면 자동으로 `ws://localhost:8765`
에 연결되어 실시간 3D 시각화가 시작됩니다.

### 방법 2 — 데모 (서버 불필요)

서버를 띄우지 않고 **`mockup_3d.html`** 만 브라우저로 열어도 됩니다.
화면의 **⚡ DEMO** 또는 **SUB-A / SUB-B / EPOC X** 버튼을 누르면
저장소에 포함된 CSV 데이터를 재생합니다.

---

## 데이터 소스 (`--source`)

| 소스 | 설명 | 추가 옵션 |
| :--- | :--- | :--- |
| `sim` | 시뮬레이션 (기본값) | — |
| `deap` | DEAP 데이터셋 재생 | `--file s01.dat --trial 0 --speed 2.0` |
| `mental` | Kaggle Mental State CSV | `--file data.csv` |
| `muse` | Muse S (muselsl/LSL) | — |
| `tgam` | NeuroSky / TGAM (시리얼) | `--serial-port COM3` 또는 `/dev/ttyUSB0` |
| `emotiv` | Emotiv EPOC X (Cortex API) | `--emotiv-id ID --emotiv-secret SECRET` |
| `openbci` | OpenBCI Ganglion(4ch)/Cyton(8ch) | `--board ganglion\|cyton --serial COM3` |

예시:

```bash
python eeg_server.py --source deap --file data/s01.dat --trial 0 --speed 2.0
python eeg_server.py --source tgam --serial-port /dev/ttyUSB0
python eeg_server.py --source openbci --board cyton --serial /dev/ttyUSB0
```

> 실제 기기(`muse` / `tgam` / `emotiv` / `openbci`)는 각각 `muselsl`·`pyserial`·Emotiv
> Cortex App·`brainflow`가 필요합니다. 미설치/미연결 시 자동으로 시뮬레이션으로
> 폴백합니다.

> 모든 소스는 v3 브릿지 호환을 위해 프레임마다 `quality`(0.0~1.0) 신호 품질 스칼라를
> 함께 전송합니다. 실험(experiment) 중 `stimulus` 마커가 발생하면 습관화 지표
> `response_amplitude` / `habituation_index`가 추가로 출력됩니다.

---

## 프로젝트 구조

```
mockup_3d.html      메인 프론트엔드 — 3D 뇌 모니터 (three.js)
eeg_server.py       메인 백엔드 — WebSocket 서버 (모든 소스 · 보정 · 실험 · ML)
├─ eeg_model.py     V-패턴 ML 감지 (EEGNet + LSTM, 룰베이스 폴백 내장)
├─ calibration.py   개인 보정 (RELAX 15s + FOCUS 15s → 임계값 자동 조정)
└─ experiment.py    실험 프로토콜 (baseline/relax/task 구간 + 행동 마커 기록)

train_model.py      ML 모델 학습 스크립트 → models/vpattern_model.pt

데이터 (CSV):
  eeg_subjecta.csv        피험자 A
  eeg_subjectb.csv        피험자 B (V-패턴 분리도 최상)
  eeg_brainwave_real.csv  A + B 결합 (486 rows)
  eeg_emotiv_epoc.csv     14채널 EPOC X 샘플 (128Hz · 6분)

tests/              pytest 테스트 모음
```

---

## V-패턴이란?

집중도가 **`thLow` 아래로 떨어졌다가 `dt`초 이내에 `thHigh` 위로 빠르게 반등**
(상승 기울기가 임계값 초과)하는 패턴입니다. "집중이 풀렸다가 다시 잡히는 순간"을
포착해 제어 신호로 활용합니다.

두 가지 감지기가 있습니다:

- **룰베이스** (`VPatternRuleBased`) — PyTorch 불필요, 항상 동작
- **머신러닝** (`EEGNetLSTM`) — 학습된 모델(`models/vpattern_model.pt`) 사용,
  모델이 없으면 자동으로 룰베이스로 폴백

---

## 의존성

**런타임**

```bash
pip install websockets numpy scipy pandas
# ML(V-패턴 학습/추론)을 쓰려면 추가로:
pip install torch
```

**개발 / 테스트**

```bash
pip install -r requirements-dev.txt   # pytest, pytest-asyncio, numpy, scipy
```

---

## 테스트

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 모델 학습 (선택)

V-패턴 ML 모델을 직접 학습시키려면:

```bash
pip install torch
python train_model.py --source deap   --file data/s01.dat
python train_model.py --source mental --file data/mental-state.csv
```

학습 결과는 `models/vpattern_model.pt`에 저장되며, `eeg_server.py`가
시작 시 자동으로 로드합니다.

---

## 라이선스

[MIT License](LICENSE) — 자유롭게 사용·수정·배포할 수 있습니다.
