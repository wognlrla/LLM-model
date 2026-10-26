"""9단계 공통: 관절 특징 계산과 GRU 동작 분류 모델 (pose_app.py, train_pose.py에서 함께 사용)"""
import numpy as np
import torch
import torch.nn as nn

# [보드 대응] GPU가 없거나(합성 실험을 CPU로 돌릴 때) 메모리가 부족하면 CPU를 쓴다
DEV = "cuda" if torch.cuda.is_available() else "cpu"

KP_CONF = 0.5
# COCO 17 관절: 0 코, 1/2 눈, 3/4 귀, 5/6 어깨, 7/8 팔꿈치, 9/10 손목, 11/12 엉덩이
NOSE, EYE_L, EYE_R, EAR_L, EAR_R, LS, RS, LW, RW, LH, RH = 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12
N_FEAT = 34 + 17 + 2      # 정규화 좌표 34 + 관절 신뢰도 17 + 정면 점수 + 이동 속도

def body(k):
    """어깨 중심과 어깨 너비. 어깨가 안 보이면 None"""
    if k[LS, 2] < KP_CONF or k[RS, 2] < KP_CONF:
        return None
    c = (k[LS, :2] + k[RS, :2]) / 2
    sw = max(abs(k[LS, 0] - k[RS, 0]), 20.0)   # 측면에서 0에 가까워지는 것 방지
    return c, sw

def facing_score(k):
    """1에 가까울수록 카메라 정면. 코·두 눈이 보이고 코가 두 귀(없으면 두 눈) 가운데 있으면 정면"""
    if k[NOSE, 2] < KP_CONF or k[EYE_L, 2] < KP_CONF or k[EYE_R, 2] < KP_CONF:
        return 0.0
    if k[EAR_L, 2] > KP_CONF and k[EAR_R, 2] > KP_CONF:
        l, r = k[EAR_L, 0], k[EAR_R, 0]
    else:
        l, r = k[EYE_L, 0], k[EYE_R, 0]
    yaw = abs(k[NOSE, 0] - (l + r) / 2) / max(abs(l - r), 1.0)   # 0 = 정면
    return float(max(0.0, 1 - 2 * yaw))

def frame_features(k, b, facing, speed):
    """한 프레임의 특징 벡터 (N_FEAT,)"""
    c, sw = b
    norm = ((k[:, :2] - c) / sw).flatten()
    return np.concatenate([norm, k[:, 2], [facing, speed]]).astype(np.float32)

class PoseGRU(nn.Module):
    def __init__(self, n_feat, n_cls, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):            # x: (B, T, F)
        _, h = self.gru(x)
        return self.fc(h[-1])

def load_model(path, device=DEV):
    ck = torch.load(path, map_location=device)
    m = PoseGRU(N_FEAT, len(ck["labels"]), ck["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    ck["model"] = m
    ck["mean"] = torch.tensor(ck["mean"], device=device)
    ck["std"] = torch.tensor(ck["std"], device=device)
    ck["device"] = device
    return ck

@torch.inference_mode()
def predict(ck, seq, device=None):
    """seq: (T, F) numpy → (라벨, 확률)"""
    device = device or ck.get("device", DEV)
    x = torch.tensor(np.asarray(seq), device=device)[None]
    x = (x - ck["mean"]) / ck["std"]
    p = torch.softmax(ck["model"](x), -1)[0]
    i = int(p.argmax())
    return ck["labels"][i], float(p[i])
