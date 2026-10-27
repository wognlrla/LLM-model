"""10단계 가상 실험: 카메라가 1대뿐이라, 가상 강의실에 가상 웹캠 2대를 두고 안내문 코드를 그대로 시험한다.

가상 강의실: 가로 8m(x) x 세로 9m(y), 원점 = 칠판 왼쪽 아래 바닥. 타겟(소화기) = (4.0, 1.5)
가상 웹캠 2대: 앞쪽 두 모서리 (0.3, 0.3, 2.4) / (7.7, 0.3, 2.4), 640x480, 렌즈 왜곡 있음, 교실 가운데를 봄
  → 타겟 주변에서 두 시선이 약 90도로 만남 (안내문 권장 60~90도)
보정도 실제 절차를 흉내 낸다:
  10-1 가상 체커보드(9x6, 25mm) 20장, 코너 검출 오차 0.25px → cv2.calibrateCamera
  10-2 기준점 10개(계산용 8 + 검증용 2), 클릭 오차 1px, 줄자 측정 오차 1cm → cv2.solvePnP (calib_extrinsics.py와 같은 계산)
사람: 3D 관절 17개(서 있음/앉음, 어깨 높이 1.35~1.50m) → 각 카메라에 투영 + 관절 오차 2px(안내문 시뮬레이션과 같은 조건)
  책상(높이 0.74m)에 가려진 관절은 신뢰도가 낮아지고, 가려진 발목은 35% 확률로 책상 앞 바닥(책상 다리)을 발목으로 잘못 잡는다.
  두 카메라 프레임 시간 차는 0~90ms 무작위 (80ms 넘으면 그 순간은 삼각측량 생략)
판정: proximity_app.py의 process_frame(), target_apply()를 그대로 호출 (PROX_OFFLINE=1이라 카메라·YOLO는 안 씀)

사용법 (보드 ~/eg2-lab/prox 에서):
  python prox_sim.py calib      # 10-1, 10-2 가상 보정 → calib/cam0.npz, cam1.npz, world_points.json
  python prox_sim.py compare    # 10-7 같은 지점 거리 비교 (ankle1 / stereo / fused)
  python prox_sim.py alarm      # 10-6 접근 알람 시나리오 5개 x 3회
  python prox_sim.py moving     # 10-8 이동 타겟(카트) TTC 알람
  python prox_sim.py plot       # 평면도 그림 (out/prox_map.png)
"""
import os, sys, json, math, collections
os.environ["PROX_OFFLINE"] = "1"
CMD = sys.argv[1] if len(sys.argv) > 1 else "compare"
sys.argv = sys.argv[:1]
import numpy as np, cv2

OUT = "out"
os.makedirs(OUT, exist_ok=True); os.makedirs("calib", exist_ok=True)
# 카메라가 1대뿐이라 기본은 '1대로 할 수 있는 것만' (ONE_CAM=0 이면 가상 2대 비교도 실행)
ONE_CAM = os.environ.get("ONE_CAM", "1") == "1"
# (모드 이름, proximity_app --mode, 설명)
MODES = ([("ankle1", "ankle1", "1대 + 발목 → 바닥"), ("상체1대", "fused", "1대 + 어깨 → 높이 1.2m 평면 추정"),
          ("mono", "mono", "1대 보완: 발목으로 잰 어깨 높이 기억")] if ONE_CAM else
         [("ankle1", "ankle1", ""), ("stereo", "stereo", ""), ("fused", "fused", "")])
W, H = 640, 480
TARGET = (4.0, 1.5)
CAM_POS = [np.array([0.3, 0.3, 2.4]), np.array([7.7, 0.3, 2.4])]
LOOK_AT = np.array([4.0, 3.6, 0.8])
K_TRUE = np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]])
DIST_TRUE = np.array([-0.28, 0.09, 0.0, 0.0, -0.012])
# 책상 (x0, x1, y0, y1), 높이 0.74m. 첫 번째는 타겟 옆 긴 책상(앉기·책상 뒤 서기 시험용)
DESKS = [(1.6, 3.6, 0.9, 1.3), (1.0, 3.3, 4.6, 5.1), (4.7, 7.0, 4.6, 5.1), (1.0, 3.3, 6.2, 6.7), (4.7, 7.0, 6.2, 6.7)]
DESK_H = 0.74
WORLD_POINTS = [
    {"id": "F1", "xyz": [2.0, 2.0, 0.0]}, {"id": "F2", "xyz": [6.0, 2.0, 0.0]},
    {"id": "F3", "xyz": [6.0, 7.5, 0.0]}, {"id": "F4", "xyz": [2.0, 7.5, 0.0]},
    {"id": "D1", "xyz": [3.3, 4.6, 0.74]}, {"id": "D2", "xyz": [4.7, 4.6, 0.74]},
    {"id": "W1", "xyz": [2.5, 9.0, 1.5]}, {"id": "W2", "xyz": [5.5, 9.0, 1.5]},
    {"id": "C1", "xyz": [4.0, 3.0, 0.0], "check": True}, {"id": "C2", "xyz": [4.7, 6.2, 0.74], "check": True},
    # 강의실 범위만 정하는 모서리 점 (room_bounds가 기준점 ±0.5m를 강의실로 보므로 앞쪽·양옆까지 포함되게). 보정 계산에는 안 씀
    {"id": "R1", "xyz": [0.3, 0.5, 0.0], "room": True}, {"id": "R2", "xyz": [7.7, 8.7, 0.0], "room": True}]


def look_at(C, target):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, (-R @ C).reshape(3, 1)


TRUE = []
for C in CAM_POS:
    R, t = look_at(C, LOOK_AT)
    TRUE.append({"R": R, "t": t, "rvec": cv2.Rodrigues(R)[0], "C": C})


def proj_true(n, X):
    """실제(참) 카메라로 3D 점들 → 왜곡 있는 픽셀, 카메라 앞인지"""
    X = np.atleast_2d(np.asarray(X, float))
    p, _ = cv2.projectPoints(X, TRUE[n]["rvec"], TRUE[n]["t"], K_TRUE, DIST_TRUE)
    depth = (TRUE[n]["R"] @ X.T + TRUE[n]["t"])[2]
    return p.reshape(-1, 2), depth > 0.1


def blocked_by_desk(C, X):
    """카메라 C에서 점 X로 가는 선분이 책상 상자(높이 0~0.74m)를 지나면 True (슬랩 방법)"""
    d = X - C
    for x0, x1, y0, y1 in DESKS:
        lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, DESK_H])
        if np.all((X >= lo) & (X <= hi)):             # 책상 밑(무릎 등)은 위에서 보는 카메라에 가려짐
            return (x0, x1, y0, y1)
        t0, t1 = 0.0, 0.999
        ok = True
        for a in range(3):
            if abs(d[a]) < 1e-9:
                if C[a] < lo[a] or C[a] > hi[a]:
                    ok = False; break
            else:
                ta, tb = sorted(((lo[a] - C[a]) / d[a], (hi[a] - C[a]) / d[a]))
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 > t1:
                    ok = False; break
        if ok:
            return (x0, x1, y0, y1)
    return None


# ---------------- 10-1, 10-2 가상 보정 ----------------
def cmd_calib():
    rng = np.random.default_rng(1)
    res = {}
    obj = np.zeros((54, 3), np.float32); obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    obj -= obj.mean(0)
    for n in range(2):
        obj_pts, img_pts = [], []
        while len(img_pts) < 20:                                   # 체커보드를 화면 곳곳·여러 각도로
            rv = rng.uniform(-0.6, 0.6, 3); rv[2] = rng.uniform(-0.4, 0.4)
            tv = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.18, 0.18), rng.uniform(0.35, 0.8)])
            p, _ = cv2.projectPoints(obj, rv, tv, K_TRUE, DIST_TRUE)
            p = p.reshape(-1, 2) + rng.normal(0, 0.25, (54, 2))
            if p.min() < 5 or p[:, 0].max() > W - 5 or p[:, 1].max() > H - 5:
                continue
            obj_pts.append(obj.astype(np.float32)); img_pts.append(p.astype(np.float32).reshape(-1, 1, 2))
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None)
        np.savez(f"calib/cam{n}_intr.npz", K=K, dist=dist, res=np.array([W, H]))
        res[f"cam{n}"] = {"rms_px": round(float(rms), 3), "f": round(float(K[0, 0]), 1), "f_true": 420.0}
    # 10-2: 기준점 클릭(1px) + 줄자(1cm) 오차로 solvePnP
    meas = {p["id"]: (np.array(p["xyz"]) + rng.normal(0, 0.01, 3) * [1, 1, 0.5]).tolist() for p in WORLD_POINTS}
    wp = [dict(p, xyz=[round(v, 3) for v in meas[p["id"]]]) for p in WORLD_POINTS]
    json.dump({"_설명": "가상 강의실 기준점 (prox_sim.py가 생성, 줄자 오차 1cm 포함)", "points": wp},
              open("calib/world_points.json", "w"), ensure_ascii=False, indent=1)
    clicks = {}
    for n in range(2):
        CP = [q for q in WORLD_POINTS if not q.get("room")]
        p, front = proj_true(n, [q["xyz"] for q in CP])
        assert front.all() and (p[:, 0] > 0).all() and (p[:, 0] < W).all() and (p[:, 1] > 0).all() and (p[:, 1] < H).all(), \
            f"cam{n}에 안 보이는 기준점 있음"
        clicks[n] = {q["id"]: p[i] + rng.normal(0, 1.0, 2) for i, q in enumerate(CP)}
    import stereo as S
    for n in range(2):
        d = np.load(f"calib/cam{n}_intr.npz"); K, dist = d["K"], d["dist"]
        ids = [q["id"] for q in WORLD_POINTS if not q.get("check") and not q.get("room")]
        o = np.float64([meas[i] for i in ids]); im = np.float64([clicks[n][i] for i in ids])
        ok, rvec, tvec = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(o, im, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        errs = np.linalg.norm(pr.reshape(-1, 2) - im, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=R, t=tvec, res=np.array([W, H]))
        C = (-R.T @ tvec).ravel()
        res[f"cam{n}"].update(mean_px=round(float(errs.mean()), 2),
                              pos_err_cm=round(float(np.linalg.norm(C - CAM_POS[n]) * 100), 1),
                              pos=C.round(2).tolist())
    c0, c1 = S.load_cam("calib/cam0.npz"), S.load_cam("calib/cam1.npz")
    ver = {}
    for q in WORLD_POINTS:
        if q.get("room"):
            continue
        i = q["id"]
        p0, p1 = S.undistort(c0, [clicks[0][i]])[0], S.undistort(c1, [clicks[1][i]])[0]
        X, _ = S.triangulate(c0, c1, p0, p1)
        ver[i] = {"check": bool(q.get("check")), "err_cm": round(float(np.linalg.norm(X - np.array(q["xyz"])) * 100), 1)}
    res["verify"] = ver
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{OUT}/calib.json", "w"), ensure_ascii=False, indent=1)


# ---------------- 사람 3D 관절과 가상 검출 ----------------
def person3d(x, y, face_deg=-90, posture="stand", sh=1.43, walk_ph=None):
    """COCO 17 관절의 3D 좌표. face_deg: 바라보는 방향(-90 = 칠판 쪽)"""
    f = np.array([math.cos(math.radians(face_deg)), math.sin(math.radians(face_deg)), 0.0])
    l = np.array([-f[1], f[0], 0.0])                       # 사람의 왼쪽
    b = np.array([x, y, 0.0]); up = np.array([0, 0, 1.0])
    if posture == "sit":
        sh = sh - 0.38
    J = np.zeros((17, 3))
    J[0] = b + 0.10 * f + (sh + 0.20) * up
    J[1], J[2] = b + 0.08 * f + 0.035 * l + (sh + 0.24) * up, b + 0.08 * f - 0.035 * l + (sh + 0.24) * up
    J[3], J[4] = b + 0.08 * l + (sh + 0.22) * up, b - 0.08 * l + (sh + 0.22) * up
    J[5], J[6] = b + 0.20 * l + sh * up, b - 0.20 * l + sh * up
    if posture == "sit":
        J[7], J[8] = b + 0.22 * l + 0.15 * f + 0.85 * up, b - 0.22 * l + 0.15 * f + 0.85 * up
        J[9], J[10] = b + 0.15 * l + 0.35 * f + 0.78 * up, b - 0.15 * l + 0.35 * f + 0.78 * up
        J[11], J[12] = b + 0.12 * l - 0.05 * f + 0.48 * up, b - 0.12 * l - 0.05 * f + 0.48 * up
        J[13], J[14] = b + 0.11 * l + 0.45 * f + 0.50 * up, b - 0.11 * l + 0.45 * f + 0.50 * up
        J[15], J[16] = b + 0.11 * l + 0.45 * f + 0.08 * up, b - 0.11 * l + 0.45 * f + 0.08 * up
    else:
        s = 0.0 if walk_ph is None else 0.18 * math.sin(walk_ph)
        J[7], J[8] = b + 0.24 * l + (sh - 0.30) * up, b - 0.24 * l + (sh - 0.30) * up
        J[9], J[10] = b + 0.25 * l + 0.05 * f + (sh - 0.58) * up, b - 0.25 * l + 0.05 * f + (sh - 0.58) * up
        J[11], J[12] = b + 0.12 * l + (sh - 0.48) * up, b - 0.12 * l + (sh - 0.48) * up
        J[13], J[14] = b + 0.11 * l + s * 0.5 * f + 0.50 * up, b - 0.11 * l - s * 0.5 * f + 0.50 * up
        J[15], J[16] = b + 0.11 * l + s * f + 0.08 * up, b - 0.11 * l - s * f + 0.08 * up
    return J


def detect(n, joints_list, rng, px_noise=2.0, false_ankle=0.35):
    """카메라 n이 본 사람들의 관절 (YOLO11n-pose 출력 흉내). 반환 (N,17,3), 사람 순서는 섞임"""
    out = []
    for J in joints_list:
        p, front = proj_true(n, J)
        conf = rng.uniform(0.72, 0.95, 17)
        inside = front & (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        for i in range(17):
            if not inside[i]:
                conf[i] = rng.uniform(0.02, 0.2)
                continue
            desk = blocked_by_desk(CAM_POS[n], J[i])
            if desk:
                conf[i] = rng.uniform(0.05, 0.3)
                if i in (15, 16) and rng.random() < false_ankle:   # 책상 다리를 발목으로 잘못 잡음
                    x0, x1, y0, y1 = desk                   # 가장 가까운 앞쪽 책상 다리(모서리)
                    fake = np.array([x0 if abs(J[i][0] - x0) < abs(J[i][0] - x1) else x1, y0, 0.0])
                    p[i] = proj_true(n, fake)[0][0]; conf[i] = rng.uniform(0.55, 0.7)
        if (conf > 0.5).sum() < 5:
            continue                                                # 사람으로 검출되지 않음
        p = p + rng.normal(0, px_noise, p.shape)
        out.append(np.concatenate([p, conf[:, None]], 1))
    rng.shuffle(out)
    return np.array(out) if out else np.zeros((0, 17, 3))


# ---------------- 실행기 ----------------
# proximity_app은 불러올 때 calib/cam*.npz를 읽으므로, 가상 보정(calib)을 먼저 해야 한다
if CMD != "calib":
    import proximity_app as pa
import stereo as S


def reset(mode, moving=False, near=1.0, far=3.0, target=TARGET):
    pa.args.mode, pa.args.moving, pa.args.near, pa.args.far = mode, moving, near, far
    pa.NUM_CAMS = 1 if mode == "ankle1" else 2
    pa.tracks.clear(); pa.Track._next = 1
    pa.events = collections.deque(maxlen=100000)
    pa.state["target"] = {"xyz": [target[0], target[1], 0.0], "how": "직접 지정", "boxes": [None, None], "scores": []}
    pa.tgt.update(kf=None, h=pa.args.target_h, last=0.0, boxes=[None, None], how="")


def run(scene, total, mode, fps=10, seed=0, moving=None, near=1.0, far=3.0, one_cam=ONE_CAM):
    """scene(t) -> [사람 3D 관절, ...]. 프레임마다 가상 검출 → process_frame. 반환: 프레임 기록, 이벤트
    one_cam=True: 카메라가 1대뿐인 실제 상황. 두 번째 카메라 입력은 비워 둔다(안내문 fused 코드가 cam0 단독 추정으로 동작)"""
    rng = np.random.default_rng(seed)
    reset(mode, moving is not None, near, far)
    T0 = 1_800_000_000.0
    log = []
    if moving:
        moving["init"](T0)
    for i in range(int(total * fps)):
        t = i / fps
        skew = rng.uniform(0, 0.09)
        k0 = detect(0, scene(t), rng)
        k1 = np.zeros((0, 17, 3)) if one_cam else detect(1, scene(t + skew), rng)
        now = T0 + t
        if moving:
            moving["step"](t, now, i, rng)
        people = pa.process_frame([k0, k1], now, skew)
        log.append((t, [dict(p) for p in people], dict(pa.state["target"])))
    ev = [(e["ts"] - T0, e["who"], e["label"]) for e in reversed(pa.events)]
    return log, ev


# ---------------- 10-7 같은 지점 비교 ----------------
U = np.array([1.0, 1.0]) / math.sqrt(2)


def find_single_view_point(rng_seed=3):
    """cam0에는 잘 보이고 cam1 화면 밖인, 타겟에서 2.5~3.5m 떨어진 지점 찾기"""
    best = None
    for x in np.arange(0.6, 7.6, 0.1):
        for y in np.arange(0.6, 8.6, 0.1):
            d = math.dist((x, y), TARGET)
            if not 2.5 <= d <= 3.5 or any(x0 - .3 <= x <= x1 + .3 and y0 - .3 <= y <= y1 + .3 for x0, x1, y0, y1 in DESKS):
                continue
            J = person3d(x, y)
            p0, f0 = proj_true(0, J); p1, f1 = proj_true(1, J)
            in0 = f0 & (p0[:, 0] > 5) & (p0[:, 0] < W - 5) & (p0[:, 1] > 5) & (p0[:, 1] < H - 5)
            in1 = f1 & (p1[:, 0] > 0) & (p1[:, 0] < W) & (p1[:, 1] > 0) & (p1[:, 1] < H)
            if in0.all() and in1.sum() < 3:              # cam1에는 사람으로 검출되지 않을 만큼만 보임
                if best is None or abs(d - 3.0) < abs(best[2] - 3.0):
                    best = (round(x, 2), round(y, 2), d)
    return best


def tests():
    T = np.array(TARGET)
    pts = [(f"{d}m, 서 있음", tuple(T + d * U), "stand") for d in (0.5, 1.0, 2.0, 3.0)]
    pts += [("1.0m, 책상에 앉음", (3.0, 1.5), "sit"), ("2.0m, 책상 뒤에 서 있음", (2.0, 1.5), "stand")]
    if ONE_CAM:                                  # 1대: 카메라에서 먼 곳의 오차 확인 (책상 줄 사이 통로)
        pts += [("6.3m, 서 있음 (카메라에서 멂)", (4.0, 7.8), "stand")]
        # 걸어 들어와서(발목 보임) 책상 뒤에 서거나 앉음(발목 가려짐): mono가 기억한 높이를 쓰는 경우
        pts += [("2.0m, 걸어와서 책상 뒤에 섬", (2.0, 1.5), "stand", (2.0, 4.0)),
                ("1.0m, 걸어와서 책상에 앉음", (3.0, 1.5), "sit", (3.0, 4.0))]
    else:
        sv = find_single_view_point()
        pts += [(f"한 카메라에만 보이는 위치 ({sv[2]:.1f}m)", sv[:2], "stand")]
    return pts


def cmd_compare():
    rows, raw = [], {}
    for item in tests():
        name, (x, y), posture = item[:3]
        walk_from = item[3] if len(item) > 3 else None
        t_arr = math.dist(walk_from, (x, y)) / 1.0 if walk_from else 0.0      # 1m/s로 걸어옴
        d_true = math.dist((x, y), TARGET)
        row = {"name": name, "d_true": round(d_true, 2)}
        for label, mode, _ in MODES:
            meds, errs, pres, kinds, poses, ids = [], [], 0, collections.Counter(), collections.Counter(), set()
            for rep in range(3):
                sh = (1.38, 1.43, 1.49)[rep]
                def scene(t, x=x, y=y, sh=sh, posture=posture, wf=walk_from, ta=t_arr):
                    if wf is not None and t < ta:                       # 걸어오는 중 (서 있는 자세)
                        r = t / ta
                        return [person3d(wf[0] + (x - wf[0]) * r, wf[1] + (y - wf[1]) * r, face_deg=-90, sh=sh, walk_ph=t * 6)]
                    return [person3d(x + 0.02 * math.sin(1.3 * t), y + 0.02 * math.cos(0.9 * t), posture=posture, sh=sh)]
                log, _ = run(scene, t_arr + 6.0, mode, seed=10 + rep)
                ds = []
                for t, people, _ in log:
                    if t < t_arr + 1.0:                                 # 도착 1초 뒤부터 잼
                        continue
                    if people:
                        p = min(people, key=lambda p: p["d"]); ds.append(p["d"]); pres += 1
                        kinds[p["kind"]] += 1; poses[p["pose"]] += 1; ids.add(p["id"])
                if ds:
                    meds.append(float(np.median(ds))); errs += [abs(v - d_true) for v in ds]
            n_frames = 3 * int(5.0 * 10)
            row[label] = {"거리": round(float(np.median(meds)), 2) if meds else None,
                         "평균오차": round(float(np.mean(errs)), 2) if errs else None,
                         "최대오차": round(float(np.max(errs)), 2) if errs else None,
                         "보인비율": round(pres / n_frames, 2), "방식": dict(kinds), "자세": dict(poses)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    json.dump(rows, open(f"{OUT}/compare.json", "w"), ensure_ascii=False, indent=1)
    print("\n| 실제 거리·자세 | " + " | ".join(f"{l} ({d})" if d else l for l, _, d in MODES) + " |\n|---" * (len(MODES) + 1) + "|")
    for r in rows:
        cell = lambda m: ("없음" if r[m]["거리"] is None else
                          f"{r[m]['거리']} (오차 평균 {r[m]['평균오차']}, 최대 {r[m]['최대오차']}, 보임 {int(r[m]['보인비율'] * 100)}%)")
        print(f"| {r['name']} (실제 {r['d_true']}m) | " + " | ".join(cell(l) for l, _, _ in MODES) + " |")


# ---------------- 10-6 접근 알람 ----------------
def path(points, speed):
    """꺾은선 경로를 일정 속도로 걷기. 반환 f(t)->(x, y, 방향), 총 시간"""
    P = [np.array(p, float) for p in points]
    seg = [np.linalg.norm(b - a) for a, b in zip(P, P[1:])]
    total = sum(seg) / speed

    def f(t):
        s = min(t * speed, sum(seg) - 1e-6)
        for a, b, L in zip(P, P[1:], seg):
            if s <= L:
                v = (b - a) / L
                return a + v * s, math.degrees(math.atan2(v[1], v[0]))
            s -= L
        return P[-1], -90
    return f, total


def cmd_alarm():
    T = np.array(TARGET)
    res = collections.OrderedDict()
    for rep in range(3):
        sp, sh, seed = (0.8, 1.0, 1.3)[rep], (1.38, 1.43, 1.49)[rep], 100 + rep
        sc = {}
        # 1) 4m에서 타겟으로 걸어옴 (0.6m에서 멈춤)
        start = T + 4.0 * U
        f, tw = path([start, T + 0.6 * U], sp)
        sc["4m에서 타겟으로 걸어옴"] = (lambda t, f=f, tw=tw: [person3d(*f(t)[0], face_deg=f(t)[1], sh=sh, walk_ph=t * 6 if t < tw else None)], tw + 2.0,
                                  lambda ev: _count(ev, "알람: 근접") == 1)     # '접근 시작'은 따로 집계(안내문 코드 특성)
        # 2) 0.5m에서 4m 밖으로 나감
        f2, tw2 = path([T + 0.5 * U, T + 4.3 * U], sp)
        sc["타겟 앞 0.5m에서 4m 밖으로 나감"] = (lambda t, f=f2, tw=tw2: [person3d(*f(max(0, t - 2))[0], face_deg=45, sh=sh, walk_ph=t * 6)], tw2 + 3.0,
                                         lambda ev: _count(ev, "알람: 이탈") == 1)
        # 3) 1.0m 경계에서 10초 서 있음 (몸이 ±5cm 흔들림)
        sc["1.0m 경계에서 10초 서 있음"] = (lambda t: [person3d(*(T + (1.0 + 0.05 * math.sin(2.1 * t)) * U), sh=sh)], 10.0,
                                     lambda ev: _count(ev, "알람: 근접") <= 1)
        # 4) 타겟 2m 옆을 지나감
        f4, tw4 = path([(0.8, 3.5 + 0.0), (7.2, 3.5)], sp)
        sc["타겟 2m 옆을 지나감"] = (lambda t, f=f4: [person3d(*f(t)[0], face_deg=0, sh=sh, walk_ph=t * 6)], tw4,
                                lambda ev: _count(ev, "알람: 근접") == 0)
        # 5) 타겟 옆 책상에 앉음 (0.8m)
        sc["타겟 옆 책상에 앉음 (0.8m)"] = (lambda t: [person3d(3.2, 1.5, posture="sit", sh=sh)], 6.0, None)
        for name, (scene, total, judge) in sc.items():
            for label, mode, _ in MODES:
                log, ev = run(scene, total, mode, seed=seed)
                evs = [e[2] for e in ev if not e[2].startswith(("등장", "퇴장"))]
                poses = collections.Counter(p["pose"] for _, ps, _ in log for p in ps)
                if judge is None:                               # 앉음: 근접 알람 (+ 2대일 때만 자세 '앉음')
                    ok = _count(evs, "알람: 근접") >= 1
                    if not ONE_CAM:
                        ok = ok and bool(poses) and poses.most_common(1)[0][0] == "앉음"
                    note = "자세는 높이를 재야 알 수 있어 1대로는 판정 불가" if ONE_CAM else f"자세 {dict(poses)}"
                else:
                    ok, note = judge(evs), ""
                res.setdefault(name, {}).setdefault(label, []).append(
                    {"ok": bool(ok), "events": evs[:6], "note": note,
                     "n": {"근접": _count(evs, "알람: 근접"), "이탈": _count(evs, "알람: 이탈"), "접근시작": _count(evs, "접근 시작")},
                     "seen": round(sum(1 for _, ps, _ in log if ps) / max(len(log), 1), 2),
                     "ids": len({p['id'] for _, ps, _ in log for p in ps})})
    print("\n| 시나리오 (3회씩) | " + " | ".join(l for l, _, _ in MODES) + " | 비고 |\n|---" * (len(MODES) + 2) + "|")
    for name, by in res.items():
        cells = []
        for label, _, _ in MODES:
            rs = by[label]
            nn = lambda k: [r["n"][k] for r in rs]
            cells.append(f"{sum(r['ok'] for r in rs)}/3 (근접 {nn('근접')}, 이탈 {nn('이탈')}, 접근시작 {nn('접근시작')})")
        first = by[MODES[0][0]][0]
        print(f"| {name} | " + " | ".join(cells) + f" | {first['note']} |")
        for label, _, _ in MODES:
            print(f"    {label} 1회차 이벤트: {', '.join(by[label][0]['events']) or '없음'}")
    json.dump(res, open(f"{OUT}/alarm.json", "w"), ensure_ascii=False, indent=1)


def _count(ev, prefix):
    return sum(1 for e in ev if e.startswith(prefix))


def _has_order(ev, prefixes):
    i = 0
    for e in ev:
        if i < len(prefixes) and e.startswith(prefixes[i]):
            i += 1
    return i == len(prefixes)


# ---------------- 10-8 이동 타겟 ----------------
def cmd_moving():
    pa.args.ttc = 2.0
    person = (2.0, 3.6)
    res = []
    hide_case = (("카메라에서 2.5초 가려짐 (1.0m/s)", 1.0, ((1.8, 4.3, "both"),)) if ONE_CAM else
                 ("한 카메라에서 가려짐 (1.0m/s)", 1.0, ((0.6, 1.8, "cam1"), (1.8, 4.3, "both"))))
    for name, v, hide in (("천천히 (0.5m/s)", 0.5, None), ("보통 (1.0m/s)", 1.0, None), ("빠르게 (1.5m/s)", 1.5, None),
                          hide_case):
        x0, x_stop = 6.0, person[0] + 0.6
        t_move = 2.0                                          # 2초 정지(타겟 찾기) 후 출발
        t_arrive = t_move + (x0 - x_stop) / v
        total = t_arrive + 3.0

        def cart_xy(t):
            if t < t_move:
                return np.array([x0, person[1]])
            return np.array([max(x_stop, x0 - v * (t - t_move)), person[1]])

        def init(T0):
            x, y = cart_xy(0)
            pa.state["target"] = {"xyz": [float(x), float(y), 0.5], "how": "삼각측량", "boxes": [None, None], "scores": []}
            pa.tgt.update(kf=S.Kalman2D((x, y)), last=T0, boxes=[None, None], how="삼각측량", h=0.5)
            pa.tgt["trail"].clear()

        def step(t, now, i, rng):
            if i % pa.args.target_every == 0:
                X = np.array([*cart_xy(t), 0.5])
                chosen = []
                for n in range(2):
                    hidden = (ONE_CAM and n == 1) or bool(hide and any(a <= t < b and (w == "both" or w == f"cam{n}") for a, b, w in hide))
                    p, front = proj_true(n, X)
                    c = p[0] + rng.normal(0, 3.0, 2)
                    ok = front[0] and 0 <= c[0] < W and 0 <= c[1] < H and not hidden
                    chosen.append(((float(c[0]), float(c[1])), [int(c[0] - 25), int(c[1] - 20), int(c[0] + 25), int(c[1] + 20)]) if ok else None)
                lost = now - pa.tgt["last"] > pa.TGT_LOST_S
                pa.target_apply(chosen, now, lost)      # 놓쳤을 때 임베딩 재식별은 '맞는 후보를 고른다'고 가정
            pa.target_publish(now)
        scene = lambda t: [person3d(*person, face_deg=0, sh=1.43)]
        log, ev = run(scene, total, "mono" if ONE_CAM else "fused", fps=7, seed=7, moving={"init": init, "step": step})
        T0 = 0
        danger = next(((t, l) for t, w, l in ev if l.startswith("알람: 충돌")), None)
        near = next(((t, l) for t, w, l in ev if l.startswith("알람: 근접")), None)
        true_d = lambda t: float(np.linalg.norm(cart_xy(t) - np.array(person)))
        sp_est = None
        if danger:
            tgt = min(log, key=lambda r: abs(r[0] - danger[0]))[2]
            sp_est = round(float(np.hypot(*tgt.get("v", [0, 0]))), 2)
        # 멈춘 뒤 거리 과소 추정 (칼만 필터가 멈춤을 늦게 반영)
        under = [true_d(t) - p["d"] for t, ps, _ in log if t_arrive <= t <= t_arrive + 2.0 for p in ps if p["d"] is not None]
        other = [l for _, w, l in ev if w == "타겟"]
        r = {"name": name, "danger": None if not danger else {"t": round(danger[0] - t_move, 2), "true_d": round(true_d(danger[0]), 2), "msg": danger[1]},
             "near": None if not near else {"t": round(near[0] - t_move, 2), "true_d": round(true_d(near[0]), 2)},
             "lead_s": round(near[0] - danger[0], 2) if (danger and near) else None, "speed_est": sp_est,
             "stop_under_m": round(max(under), 2) if under else None, "target_events": other,
             "lost_frames": sum(1 for _, _, tg in log if tg.get("lost")),
             "pred_frames": sum(1 for _, _, tg in log if "(예측)" in tg.get("how", ""))}
        res.append(r)
        print(json.dumps(r, ensure_ascii=False))
    json.dump(res, open(f"{OUT}/moving.json", "w"), ensure_ascii=False, indent=1)


# ---------------- 평면도 ----------------
def cmd_plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.add_patch(plt.Rectangle((0, 0), 8, 9, fill=False, ec="#888"))
    for x0, x1, y0, y1 in DESKS:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#c8a96e", alpha=0.6))
    for n, C in enumerate(CAM_POS):
        ax.plot(*C[:2], "s", color="#2a7fff"); ax.annotate(f"cam{n}", C[:2], (C[0] + 0.15, C[1] + 0.2), color="#2a7fff")
        ax.plot([C[0], LOOK_AT[0]], [C[1], LOOK_AT[1]], "--", color="#2a7fff", lw=0.8)
    ax.plot(*TARGET, "D", color="#d0f", ms=9); ax.annotate("타겟", TARGET, (TARGET[0] + 0.15, TARGET[1] - 0.3), color="#d0f")
    for r, c in ((1.0, "#f55"), (3.0, "#999")):
        ax.add_patch(plt.Circle(TARGET, r, fill=False, ls="--", ec=c))
    for item in tests():
        name, (x, y), posture = item[:3]
        ax.plot(x, y, "o", color="#2b2" if posture == "stand" else "#e80")
        ax.annotate(name.split(",")[0].split(" (")[0], (x, y), (x + 0.12, y + 0.12), fontsize=8)
    ax.set_xlim(-0.3, 8.3); ax.set_ylim(9.3, -0.3); ax.set_aspect("equal")
    ax.set_xlabel("x (m, 칠판 방향)"); ax.set_ylabel("y (m, 교실 뒤쪽)")
    ax.set_title("가상 강의실 배치 (위에서 본 모습)")
    fig.tight_layout(); fig.savefig(f"{OUT}/prox_map.png", dpi=110)
    print(f"저장: {OUT}/prox_map.png")


if __name__ == "__main__":
    {"calib": cmd_calib, "compare": cmd_compare, "alarm": cmd_alarm, "moving": cmd_moving, "plot": cmd_plot}[CMD]()
