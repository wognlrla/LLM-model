# Jetson 멀티모달 임베딩 실습 (EmbeddingGemma 2)

Jetson Orin Nano Super 보드 한 대에서 `google/embeddinggemma-2`를 돌려 본 실습 기록입니다(team2).
- **검색**: 텍스트·이미지·동영상·음성
- **미세 조정**: 강의실 도메인 데이터로 추가 학습
- **실시간 검색**: 카메라 영상에서 물체를 검출하고, 웹에서 글자로 검색

👉 **자세한 결과는 [결과 보고서](docs/결과보고서.md)에 있습니다.**

## 핵심 결과

| 단계 | 결과 |
|---|---|
| 3 텍스트 검색 | 768·256·128차원 모두 2/2 정답, 질의 2개 165ms |
| 4 영상 검색 | 카메라 장면 사진 1/3 정답(△1), 이미지 5장 2.4초 |
| 5 음성 검색 | 2/3 정답(합성 음성만 오답), WAV 3개 0.35초 |
| 6 성능 | 전체 모델(744M)도 동작, 실행 중 최대 RAM 6.4GB / 7.3GB |
| 7 미세 조정 | accuracy@1 0.833 → 0.85, accuracy@3 0.917 → 0.967. 일반 질의 1개는 나빠짐(과적합 신호) |
| 8 실시간 검색 | yolo는 세탁기를 clock·cell phone으로 오인식. relabel이 드럼·통돌이 세탁기로 교정(3/4). yoloe만 기본 기준에서 세탁기·상자 검출 → yoloe 박스 + relabel 라벨 조합 권장 |

## 실행 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin Nano Super, JetPack 7.2.1, CUDA 13.2, Python 3.12, 전력 모드 MAXN_SUPER, 스왑 8GB |
| 패키지 | torch 2.14.1+cu132, sentence-transformers 6.2.0, transformers 5.20.1, ultralytics 8.4 |
| 카메라 | TP-Link VIGI C540-W (보드 유선랜 직결, RTSP) |

## 코드 (`code/`)

| 파일 | 단계 | 내용 |
|---|---|---|
| `check.py` | 2 | 설치 확인 (torch GPU, 버전) |
| `eg2.py` | 2~ | 공통 로더 + 보드용 동영상 함수(`video_frames`, `extract_frames`) |
| `test_text.py` | 3 | 텍스트 검색, 차원별 지연 |
| `capture.py`, `snap.py` | 4 | 카메라 녹화(10초 등), 사진 한 장 찍기 |
| `test_vision.py` | 4 | 이미지·동영상 검색 |
| `make_audio.py` | 5 | 경보음·잡음 생성, 카메라 마이크 녹음, 16kHz 모노 변환 |
| `test_audio.py` | 5 | 음성 검색 |
| `bench.py`, `bench_results.txt` | 6 | 5가지 구성 성능 측정 스크립트와 결과 |
| `make_data.py`, `data/` | 7 | 강의실 도메인 학습 282쌍·평가 60쌍 생성 |
| `eval_ir.py`, `train_ft.py` | 7 | 평가(Recall@k)·미세 조정 |
| `baseline_eval.txt`, `finetuned_eval.txt` | 7 | 학습 전후 평가 결과 |
| `app.py`, `run_app.sh` | 8 | 실시간 물체 검출·검색 웹앱(yolo / relabel / yoloe), 실행 스크립트 |
| `compare_photos.py`, `compare_results/` | 8 | 저장한 같은 사진으로 세 모드 비교, 모드별 결과(JSON) |
| `guard_run.sh`, `mem_guard.sh` | 8 | 보드 메모리가 400MB 아래로 떨어지면 실습 프로그램만 끄는 보호 스크립트 |

### 실행 예

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
python test_text.py                     # 3단계
python capture.py 10                    # 카메라 10초 녹화 + 프레임 추출
python test_vision.py                   # 4단계
python bench.py full                    # 6단계 (text | vision_img | vision_vid | audio | full)
python eval_ir.py && python train_ft.py && python eval_ir.py models/eg2-classroom   # 7단계
EXTRA_LABELS=1 ./run_app.sh relabel --flip   # 8단계, 브라우저에서 :8000
```

## 안내문과 다르게 한 것

이 보드에서는 안내문 그대로 하면 막히는 곳이 있어서 다르게 처리했습니다. 자세한 내용은 [결과 보고서](docs/결과보고서.md)의 마지막 표에 있습니다.
- **torchaudio·torchcodec 제거**: 보드의 CUDA·ffmpeg와 맞지 않아 import가 실패합니다. 동영상은 PyAV, 음성은 librosa로 읽습니다.
- **보드 ffmpeg 기능 부족**: 녹화·RTSP·소리 생성이 안 돼서, 해당 기능을 파이썬 스크립트로 대신했습니다.
- **USB 웹캠·마이크 대신 VIGI 카메라 사용**: 앱에 `--source`, `--flip` 옵션을 추가했고, 최신 프레임만 쓰도록 했습니다.

## 올리지 않은 것

- **카메라 계정 정보**: `.camera_url`
- **카메라 사진·영상**: 사람과 이름표가 찍혀 있습니다.
- **모델 파일**: 미세 조정 모델, YOLO 가중치
- **로그**
