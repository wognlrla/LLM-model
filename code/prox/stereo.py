"""10단계 공통: 카메라 보정값 로드, 왜곡 보정, 삼각측량, 단일 카메라 높이 평면 추정
좌표계: 강의실 바닥 기준. x = 칠판과 평행(오른쪽 +), y = 교실 뒤쪽(+), z = 높이(위 +), 단위 m
"""
import json
import numpy as np
import cv2

KP_CONF = 0.5
NOSE, LS, RS, LA, RA = 0, 5, 6, 15, 16

def load_cam(path):
    d = np.load(path)
    K, dist, R, t = d["K"], d["dist"], d["R"], d["t"].reshape(3, 1)
    return {"K": K, "dist": dist, "R": R, "t": t, "P": K @ np.hstack([R, t]),
            "C": (-R.T @ t).ravel(), "Kinv": np.linalg.inv(K)}

def load_world(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["points"]

def room_bounds(points, pad=0.5):
    xy = np.array([p["xyz"][:2] for p in points], dtype=float)
    return [float(xy[:, 0].min() - pad), float(xy[:, 1].min() - pad),
            float(xy[:, 0].max() + pad), float(xy[:, 1].max() + pad)]

def undistort(cam, pts):
    """왜곡이 있는 픽셀 좌표 → 이상적인(왜곡 없는) 픽셀 좌표 (N, 2)"""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPoints(pts, cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)

def project(cam, X):
    """3D 점 → 왜곡 없는 픽셀 좌표"""
    x = cam["P"] @ np.append(np.asarray(X, float), 1.0)
    return x[:2] / x[2]

def project_distorted(cam, X):
    """3D 점 → 실제 영상 픽셀 좌표 (화면 표시용)"""
    rvec, _ = cv2.Rodrigues(cam["R"])
    p, _ = cv2.projectPoints(np.float64([X]), rvec, cam["t"], cam["K"], cam["dist"])
    return p.reshape(2)

def triangulate(c0, c1, p0, p1):
    """두 카메라의 왜곡 없는 픽셀 좌표 → 3D 점, 평균 재투영 오차(px)"""
    X = cv2.triangulatePoints(c0["P"], c1["P"], np.float64(p0).reshape(2, 1), np.float64(p1).reshape(2, 1))
    X = (X[:3] / X[3]).ravel()
    err = (np.linalg.norm(project(c0, X) - p0) + np.linalg.norm(project(c1, X) - p1)) / 2
    return X, float(err)

def ray_plane(cam, p, h):
    """카메라 1대: 픽셀(왜곡 없음)을 지나는 시선이 높이 h 평면과 만나는 점"""
    d = cam["R"].T @ (cam["Kinv"] @ np.array([p[0], p[1], 1.0]))
    C = cam["C"]
    if abs(d[2]) < 1e-6:
        return None
    s = (h - C[2]) / d[2]
    return C + s * d if s > 0 else None

def height_on_vertical(cam, p, xy):
    """[1대 보완] 픽셀(왜곡 없음)을 지나는 시선이 바닥 점 xy 위 수직선에 가장 가까워지는 높이.
    발목으로 바닥 위치를 알 때, 같은 사람의 어깨가 몇 m 높이에 있는지 재는 데 쓴다."""
    d = cam["R"].T @ (cam["Kinv"] @ np.array([p[0], p[1], 1.0]))
    C = cam["C"]
    dxy = d[:2]
    s = float((np.asarray(xy, float) - C[:2]) @ dxy / max(float(dxy @ dxy), 1e-9))
    return float(C[2] + s * d[2]) if s > 0 else None

def upper_body(k):
    """어깨 중심, 어깨가 안 보이면 코. 책상에 가려지지 않는 관절"""
    if k[LS, 2] > KP_CONF and k[RS, 2] > KP_CONF:
        return (k[LS, :2] + k[RS, :2]) / 2
    if k[NOSE, 2] > KP_CONF:
        return k[NOSE, :2].copy()
    return None

def ankle(k):
    """보이는 발목의 평균 (10-3 문제 확인용)"""
    vis = [k[i, :2] for i in (LA, RA) if k[i, 2] > KP_CONF]
    return np.mean(vis, axis=0) if vis else None

def match_people(c0, c1, pts0, pts1, max_err=15.0, h_range=(0.5, 2.1)):
    """두 카메라의 사람(왜곡 없는 상체 좌표)을 재투영 오차가 작은 순으로 1:1 매칭
    반환: [(i, j, X(3,), err)], 매칭 안 된 cam0 인덱스 집합, cam1 인덱스 집합"""
    cands = []
    for i, a in enumerate(pts0):
        for j, b in enumerate(pts1):
            X, err = triangulate(c0, c1, a, b)
            if err <= max_err and h_range[0] <= X[2] <= h_range[1]:
                cands.append((err, i, j, X))
    cands.sort(key=lambda c: c[0])
    used0, used1, out = set(), set(), []
    for err, i, j, X in cands:
        if i in used0 or j in used1:
            continue
        out.append((i, j, X, err))
        used0.add(i); used1.add(j)
    return out, set(range(len(pts0))) - used0, set(range(len(pts1))) - used1

# ---------------- 10-8 이동 타겟: 칼만 필터와 상대 운동 ----------------
class Kalman2D:
    """등속 모델 칼만 필터. 상태 (x, y, vx, vy), 측정 (x, y). q: 가속 잡음, r: 측정 잡음(m²)"""
    def __init__(self, xy, q=1.0, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([r, r, 1.0, 1.0])
        self.q, self.r = q, r

    def predict(self, dt):
        if dt <= 0:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * G @ G.T

    def update(self, z):
        Hm = np.eye(2, 4)
        S_ = Hm @ self.P @ Hm.T + self.r * np.eye(2)
        K = self.P @ Hm.T @ np.linalg.inv(S_)
        self.x = self.x + K @ (np.asarray(z, float) - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def at(self, dt):
        """dt초 뒤 예측 위치 (상태는 바꾸지 않음)"""
        return self.x[:2] + self.x[2:] * dt

def relative_state(p, t):
    """p, t: 상태 벡터 (x, y, vx, vy). 반환: 거리(m), 접근 속도(m/s, +면 가까워짐), TTC(초 또는 None)"""
    dp, dv = t[:2] - p[:2], t[2:] - p[2:]
    d = float(np.linalg.norm(dp))
    closing = -float(dp @ dv) / max(d, 1e-6)
    ttc = d / closing if closing > 0.1 else None
    return d, closing, ttc
