"""9단계 합성 실험: 사람 영상 대신 'YOLO-pose가 내놓았을 관절 좌표'를 직접 만들어 판정 로직을 시험한다.

- 관절 17개(COCO), 1280x720 화면, 15fps 기준 (8단계에서 이 보드 검출 속도가 약 15fps였음)
- 체격(어깨 너비 55~120px), 몸 방향, 머리 방향, 관절 흔들림(1.5~2.5px), 손목 가림(3%),
  팔 올리고 내리는 0.4초 전환, 측면일 때 먼 팔 가림을 넣어 실제 검출 결과처럼 만든다.
- 판정은 pose_app.py의 step()을 그대로 호출한다 (POSE_OFFLINE=1이라 카메라·YOLO는 안 씀).

사용법 (보드 ~/eg2-lab/pose 에서):
  python pose_sim.py rules        # 9-1 raw / 9-2 smooth 동작 횟수 비교 (9-4 표)
  python pose_sim.py behavior     # 9-5 행동 패턴 4가지 x 3회
  python pose_sim.py record       # 9-3 학습 데이터 CSV 생성 (data/action, data/behavior)
  python pose_sim.py evaluate     # 9-6 규칙 vs 학습 모델, 학습에 없던 '낯선 사람'으로 비교
  python pose_sim.py video        # 9-2 시나리오를 뼈대 영상(out/pose_demo.mp4)으로 저장
"""
import os, sys, json, math, csv, collections
os.environ["POSE_OFFLINE"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # 작은 GRU라 CPU로 충분 (같은 보드의 다른 GPU 작업 보호)
CMD = sys.argv[1] if len(sys.argv) > 1 else "rules"
sys.argv = sys.argv[:1]                               # pose_app 옵션은 아래에서 직접 바꾼다
import numpy as np
import pose_app as pa
from pose_model import load_model

FPS, DT = 15, 1.0 / 15
W, H = 1280, 720
RAMP = 0.4                     # 팔 올리고 내리는 시간(초)
OUT = "out"
os.makedirs(OUT, exist_ok=True)

# 팔꿈치·손목 위치 (어깨 관절 기준, 어깨 너비 단위, x는 바깥쪽)
ARM_DOWN = ((0.08, 0.80), (0.12, 1.55))
ARM_UP = ((0.20, -0.65), (0.25, -1.45))
HEAD = {0: (0.0, 0.28, -0.85), 1: (0.13, 0.20, -0.97), 2: (-0.13, 0.20, -0.97),
        3: (0.32, -0.02, -0.92), 4: (-0.32, -0.02, -0.92)}          # (x, z, y)
LOWER = {11: (0.32, 1.9), 12: (-0.32, 1.9), 13: (0.33, 2.9), 14: (-0.33, 2.9),
         15: (0.33, 3.9), 16: (-0.33, 3.9)}


def lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def make_kps(s, t, rng):
    """사람 상태 s → 관절 (17,3) [x, y, 신뢰도]. s의 주요 키:
    cx, cy(어깨 중심 px), sw(실제 어깨 너비 px), yaw(몸 방향 도), head(머리 방향 도),
    ul, ur(왼/오른팔 올림 0~1), wave('l'/'r'), wf, wa(흔드는 빈도·폭), act('clap'/'cross'),
    walk(걷는 중), jit(흔들림 px), wrist_override({9:(x,y),10:(x,y)} px)"""
    sw, cb = s["sw"], math.cos(math.radians(s.get("yaw", 0)))
    cx, cy = s["cx"], s["cy"]
    P = np.zeros((17, 2)); C = rng.uniform(0.82, 0.97, 17)
    X = lambda x: cx + x * sw * cb
    Y = lambda y: cy + y * sw
    phi = math.radians(s.get("head", 0))
    for i, (x, z, y) in HEAD.items():
        P[i] = (cx + (x * math.cos(phi) + z * math.sin(phi)) * sw, Y(y))
    deg = s.get("head", 0)
    if abs(deg) > 55:                                  # 먼 쪽 귀가 안 보임
        C[4 if deg > 0 else 3] = 0.25
    if abs(deg) > 75:                                  # 먼 쪽 눈도 안 보임
        C[2 if deg > 0 else 1] = 0.3
    P[5], P[6] = (X(0.5), cy), (X(-0.5), cy)
    for i, (x, y) in LOWER.items():
        dx = 0.15 * math.sin(2 * math.pi * 1.8 * t) * (1 if i % 2 else -1) if (s.get("walk") and i >= 15) else 0
        P[i] = (X(x + dx), Y(y))
    for side, sg, ei, wi, u in (("l", 1, 7, 9, s.get("ul", 0)), ("r", -1, 8, 10, s.get("ur", 0))):
        (ex, ey), (wx, wy) = lerp(ARM_DOWN[0], ARM_UP[0], u), lerp(ARM_DOWN[1], ARM_UP[1], u)
        ex, wx = sg * (0.5 + ex), sg * (0.5 + wx)
        if s.get("wave") == side and u > 0.9:
            ph = math.sin(2 * math.pi * s.get("wf", 2.2) * t)
            wx += s.get("wa", 0.38) * ph; ex += 0.12 * ph
        P[ei], P[wi] = (X(ex), Y(ey)), (X(wx), Y(wy))
    act = s.get("act")
    if act == "clap":                                  # 가슴 앞에서 손을 모았다 벌렸다
        o = 0.5 + 0.5 * math.cos(2 * math.pi * s.get("cf", 2.5) * t)
        P[9], P[10] = (X(0.10 + 0.16 * o), Y(0.85)), (X(-0.10 - 0.16 * o), Y(0.85))
        P[7], P[8] = (X(0.45), Y(0.75)), (X(-0.45), Y(0.75))
    elif act == "cross":                               # 팔짱: 손목이 반대쪽 팔꿈치 근처, 일부 가려짐
        P[9], P[10] = (X(-0.32), Y(0.78)), (X(0.32), Y(0.72))
        P[7], P[8] = (X(0.52), Y(0.82)), (X(-0.52), Y(0.82))
        C[9], C[10] = rng.uniform(0.5, 0.72, 2)
    for i, xy in s.get("wrist_override", {}).items():
        P[i] = xy
        P[i - 2] = ((P[i - 4][0] + xy[0]) / 2, (P[i - 4][1] + xy[1]) / 2 + 0.15 * sw)   # 팔꿈치는 어깨와 손목 사이
    # 측면이면 먼 팔(화면 기준 뒤쪽)이 자주 가려짐
    if abs(s.get("yaw", 0)) > 50:
        far_w, far_e = (10, 8) if s["yaw"] > 0 else (9, 7)
        if rng.random() < 0.6:
            C[far_w], C[far_e] = 0.3, 0.4
    for wi in (9, 10):                                 # 가끔 손목이 잠깐 안 잡힘
        if rng.random() < 0.03:
            C[wi] = 0.35
    for i in s.get("wrist_override", {}):              # 악수 중 맞잡은 손은 두 사람 사이에서 보임
        C[i] = rng.uniform(0.75, 0.92)
    jit = s.get("jit", 1.5)
    noise = rng.normal(0, jit, (17, 2))
    noise[7:11] *= 1.7                                 # 팔·손목은 더 흔들림
    P += noise
    vis = P[C > 0.5]
    box = [int(vis[:, 0].min() - 20), int(vis[:, 1].min() - 20), int(vis[:, 0].max() + 20), int(vis[:, 1].max() + 20)]
    return np.concatenate([P, C[:, None]], 1).astype(np.float32), box


# ---------------- 동작 순서 (프로그램) ----------------
POSE_U = {"idle": (0, 0), "both": (1, 1), "one": (0, 1), "wave": (0, 1), "clap": (0, 0), "cross": (0, 0)}


def program(segments, base):
    """segments: [(동작, 초)] → t를 받아 상태를 돌려주는 함수 (전환 구간은 RAMP초 동안 선형 이동)"""
    starts, t0 = [], 0.0
    for name, d in segments:
        starts.append((t0, t0 + d, name)); t0 += d
    total = t0

    def f(t):
        if t >= total:
            return None
        i = next(j for j, (a, b, _) in enumerate(starts) if a <= t < b)
        a, _, name = starts[i]
        prev = starts[i - 1][2] if i else name
        k = min(1.0, (t - a) / RAMP)
        pu, cu = POSE_U[prev], POSE_U[name]
        s = dict(base, ul=pu[0] + (cu[0] - pu[0]) * k, ur=pu[1] + (cu[1] - pu[1]) * k)
        if name == "wave":
            s["wave"] = "r"
        if name in ("clap", "cross"):
            s["act"] = name
        s["truth"] = name
        return s
    return f, total


def run(people, total, record=None, pair=None, seed=0, frame_log=None, tail=4.0, t_offset=0.0):
    """people: {tid: f(t)->상태}. pose_app.step()에 프레임마다 넣는다. tail초 빈 화면으로 추적을 닫는다.
    t_offset: 녹화를 이어 붙일 때 시각을 밀어 주는 값 (같은 사람이 끊김 없이 이어서 움직인 것처럼)"""
    rng = np.random.default_rng(seed)
    for d in (pa.tracks, pa.pairs, pa.counts, pa.beh_counts):
        d.clear()
    pa.pair_on.clear(); pa.events.clear()
    pa.rec = csv.writer(record) if record else None
    T0 = 1_800_000_000.0 + t_offset
    n = int((total + tail) * FPS)
    for i in range(n):
        t = i * DT
        states = {tid: f(t) for tid, f in people.items()}
        states = {k: v for k, v in states.items() if v is not None}
        if pair:
            pair(states, t)
        ids, kps, boxes = [], [], []
        for tid, s in states.items():
            k, b = make_kps(s, t, rng)
            ids.append(tid); kps.append(k); boxes.append(b)
        pa.step(ids, np.array(kps) if kps else [], boxes, T0 + t)
        if frame_log is not None:
            for tid, s in states.items():
                st = pa.tracks.get(tid)
                if st:
                    frame_log.append((t, tid, s.get("truth"), st["label"], st["bstate"], s, kps[ids.index(tid)]))
    return dict(pa.counts), dict(pa.beh_counts), list(pa.events)


# ---------------- 9-1 / 9-2: 동작 횟수 ----------------
FRONT = dict(cx=640, cy=260, sw=95, yaw=10, head=5, jit=1.6, wf=2.2, wa=0.38)
SIDE = dict(FRONT, yaw=65, head=60)


def rep(name, n=3, hold=3.0, rest=3.0):
    return [("idle", 3.0)] + sum([[(name, hold), ("idle", rest)] for _ in range(n)], [])


def handshake_scene(n=3, seed=0, gap_on=0.05):
    """두 사람이 비스듬히(몸 방향 ±60도) 마주 서서 악수 n번: 손 뻗기 0.6초 → 3초 유지 → 손 내리기"""
    rng = np.random.default_rng(seed + 100)
    a = dict(cx=520, cy=270, sw=100, yaw=-60, head=80, jit=1.6, truth="idle")      # 왼쪽 사람: 오른쪽을 봄
    b = dict(cx=520 + 1.9 * 100, cy=275, sw=98, yaw=60, head=-80, jit=1.6, truth="idle")  # 오른쪽 사람: 왼쪽을 봄
    cycle = 6.6
    total = 2.0 + n * cycle

    def pair(states, t):
        tt = t - 2.0
        if tt < 0:
            return
        c, ph = divmod(tt, cycle)
        if c >= n:
            return
        A, B = states[1], states[2]
        xm, ym = (A["cx"] + B["cx"]) / 2, A["cy"] + 0.9 * A["sw"] + 6 * math.sin(2 * math.pi * 2 * t)
        if ph < 0.6:
            reach = ph / 0.6; gap = 1.2 * A["sw"] * (1 - reach) + gap_on * A["sw"]
        elif ph < 3.6:
            reach, gap = 1.0, gap_on * A["sw"]
        elif ph < 4.2:
            reach = 1 - (ph - 3.6) / 0.6; gap = 1.2 * A["sw"] * (1 - reach) + gap_on * A["sw"]
        else:
            return
        wa = (xm - gap / 2, ym); wb = (xm + gap / 2, ym)
        # 손을 뻗지 않았을 때 위치와 섞어 자연스럽게 이동
        A["wrist_override"] = {10: wa}
        B["wrist_override"] = {10: wb}
        A["truth"] = B["truth"] = "shake" if ph < 3.6 and ph >= 0.6 else "reach"
    people = {1: lambda t: dict(a) if t < total else None, 2: lambda t: dict(b) if t < total else None}
    return people, total, pair


def cmd_rules():
    rows = []
    scen = [("만세 (3회)", rep("both"), FRONT), ("한팔들기 (3회)", rep("one"), FRONT),
            ("손흔들기 (3회)", rep("wave"), FRONT), ("측면에서 만세 (3회)", rep("both"), SIDE)]
    for mode in ("raw", "smooth"):
        pa.MODE = mode; pa.args.behavior = False; pa.models = []; pa.SEQ_LEN = 1
        for title, seg, base in scen:
            f, total = program(seg, base)
            c, _, _ = run({1: f}, total, seed=1)
            rows.append((title, mode, c))
        people, total, pair = handshake_scene()
        c, _, _ = run(people, total, pair=pair, seed=1)
        rows.append(("악수 (3회)", mode, c))
    table = collections.OrderedDict()
    for title, mode, c in rows:
        table.setdefault(title, {})[mode] = c
    print("\n| 동작 (실제 횟수) | 9-1 raw 카운트 | 9-2 smooth 카운트 |\n|---|---|---|")
    fmt = lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1])) or "없음"
    for title, d in table.items():
        print(f"| {title} | {fmt(d['raw'])} | {fmt(d['smooth'])} |")
    json.dump(table, open(f"{OUT}/rules.json", "w"), ensure_ascii=False, indent=1)


# ---------------- 9-5: 행동 패턴 ----------------
def walker(kind, seed, sw=90, speed=2.6, look=2.0, stop=5.0, x0=40, cy=300, linger=15.0):
    """kind: passing / glance / interested / linger. 반환: (f(t), 총 시간)"""
    rng = np.random.default_rng(seed)
    v = speed * sw                                       # px/초 (어깨 너비/초 x 어깨 너비)
    side_head = rng.uniform(70, 85)
    base = dict(cy=cy, sw=sw, yaw=80, jit=1.5)
    if kind in ("passing", "glance"):
        total = (W - 2 * x0) / v
        t_look = total / 2 - look / 2

        def f(t):
            if t >= total:
                return None
            head = 3.0 if (kind == "glance" and t_look <= t < t_look + look) else side_head
            return dict(base, cx=x0 + v * t, head=head, walk=True, truth=kind)
    elif kind == "interested":
        walk_in = 2.0; total = walk_in + stop + 2.0

        def f(t):
            if t >= total:
                return None
            if t < walk_in:
                return dict(base, cx=x0 + v * t, head=side_head, walk=True, truth=kind)
            x_stop = x0 + v * walk_in
            if t < walk_in + stop:
                return dict(base, cx=x_stop, head=rng.normal(3, 2), yaw=rng.normal(5, 2), truth=kind)
            return dict(base, cx=x_stop + v * (t - walk_in - stop), head=side_head, walk=True, truth=kind)
    else:                                                # linger: 옆을 보며 서 있음
        total = linger

        def f(t):
            if t >= total:
                return None
            return dict(base, cx=500 + 4 * math.sin(t), head=side_head, yaw=60, truth=kind)
    return f, total


def cmd_behavior():
    pa.MODE = "smooth"; pa.args.behavior = True; pa.models = []; pa.SEQ_LEN = 1
    expect = {"passing": "지나감", "glance": "힐끗 봄", "interested": "관심 있음", "linger": "머무름"}
    title = {"passing": "그냥 지나감", "glance": "걸으며 2초 봄", "interested": "멈춰서 5초 봄", "linger": "옆을 보며 15초 서 있음"}
    res = {}
    for kind in expect:
        outs = []
        for r in range(3):
            f, total = walker(kind, seed=10 + r, sw=80 + 12 * r, speed=2.3 + 0.4 * r)
            _, beh, ev = run({1: f}, total, seed=20 + r)
            outs.append(next(iter(beh), "관찰 중"))
        res[kind] = outs
    print("\n| 시나리오 (3회씩) | 기대 패턴 | 맞은 횟수 | 나온 패턴 |\n|---|---|---|---|")
    for kind, outs in res.items():
        ok = sum(o == expect[kind] for o in outs)
        print(f"| {title[kind]} | {expect[kind]} | {ok}/3 | {', '.join(outs)} |")
    json.dump(res, open(f"{OUT}/behavior.json", "w"), ensure_ascii=False, indent=1)


# ---------------- 9-3: 학습 데이터 ----------------
TRAIN_PEOPLE = [dict(cx=600, cy=260, sw=80, yaw=0, head=0, jit=1.5, wf=2.2, wa=0.38, cf=2.5),
                dict(cx=700, cy=280, sw=95, yaw=15, head=10, jit=1.8, wf=2.0, wa=0.34, cf=2.2),
                dict(cx=560, cy=250, sw=105, yaw=-20, head=-12, jit=1.6, wf=2.5, wa=0.42, cf=2.8)]
ACTIONS = {"idle": "idle", "wave": "wave", "clap": "clap", "arms_crossed": "cross"}


# 2차: 안내문 조언대로 '사람·각도·거리를 바꿔 가며' 더 녹화하고, 모델이 규칙을 덮어쓰지 않게 만세(arms_up) 라벨도 추가
TRAIN_PEOPLE_V2 = TRAIN_PEOPLE + [
    dict(cx=900, cy=320, sw=60, yaw=32, head=22, jit=2.3, wf=1.8, wa=0.30, cf=2.0),
    dict(cx=420, cy=230, sw=122, yaw=-35, head=-25, jit=1.3, wf=2.6, wa=0.44, cf=3.1),
    dict(cx=760, cy=290, sw=72, yaw=-10, head=-5, jit=2.0, wf=1.7, wa=0.26, cf=1.7)]
ACTIONS_V2 = dict(ACTIONS, arms_up="both")


def cmd_record(people=TRAIN_PEOPLE, actions=ACTIONS, root="data"):
    pa.MODE = "smooth"; pa.args.behavior = False; pa.models = []; pa.SEQ_LEN = 1
    os.makedirs(f"{root}/action", exist_ok=True); os.makedirs(f"{root}/behavior", exist_ok=True)
    for label, act in actions.items():
        with open(f"{root}/action/{label}.csv", "w", newline="") as fh:
            for i, base in enumerate(people):
                f, total = program([(act, 35.0)], base)
                run({100 + i: f}, total, record=fh, seed=300 + i, tail=0.5)
    # 행동: 한 사람이 카메라 앞을 12번 왔다 갔다 하는 녹화 (끊김 없이 이어 붙여 하나의 긴 시퀀스가 되게 함).
    # 지나가는 한 번은 6~11초로 짧아서, 따로 끊으면 train_pose.py의 '뒷부분 20% 검증'에 80프레임 윈도가 안 나온다.
    for li, label in enumerate(("passing", "glance", "interested")):
        with open(f"{root}/behavior/{label}.csv", "w", newline="") as fh:
            off = 0.0
            for r in range(12):
                f, total = walker(label, seed=500 + r, sw=75 + 3 * r, speed=1.4 + 0.05 * r, look=1.6 + 0.08 * r,
                                  cy=280 + 5 * (r % 4))
                run({200 + li: f}, total, record=fh, seed=600 + r, tail=0, t_offset=off)
                off += total + DT
    for d in (f"{root}/action", f"{root}/behavior"):
        for p in sorted(os.listdir(d)):
            print(f"{d}/{p}: {sum(1 for _ in open(f'{d}/{p}'))}줄")


# ---------------- 9-6: 규칙 vs 학습 모델 (낯선 사람) ----------------
NEW_PEOPLE = [dict(cx=820, cy=300, sw=62, yaw=35, head=25, jit=2.2, wf=1.6, wa=0.28, cf=1.9),     # 멀리, 비스듬히, 천천히 작게 흔듦
              dict(cx=480, cy=240, sw=118, yaw=-28, head=-20, jit=1.4, wf=2.7, wa=0.45, cf=3.0)]  # 가까이, 반대쪽으로 비스듬히
EXPECT = {"idle": {"기본"}, "wave": {"손흔들기", "wave"}, "clap": {"clap"}, "cross": {"arms_crossed"},
          "both": {"만세", "arms_up"}}


def frame_acc(log, warm=1.0):
    """프레임마다 화면 라벨이 정답인지 (각 동작 시작 후 warm초는 반응 시간이라 제외)"""
    seg_start, last, hit, tot = {}, {}, collections.Counter(), collections.Counter()
    for t, tid, truth, label, _, _, _ in log:
        if last.get(tid) != truth:
            seg_start[tid], last[tid] = t, truth
        if t - seg_start[tid] < warm or truth not in EXPECT:
            continue
        tot[truth] += 1; hit[truth] += label in EXPECT[truth]
    return {k: round(hit[k] / tot[k], 3) for k in tot}


def cmd_evaluate(mdir="models", tag=""):
    act_m, beh_m = load_model(f"{mdir}/action.pt"), load_model(f"{mdir}/behavior.pt")
    seg = [("idle", 8), ("wave", 8), ("idle", 6), ("clap", 8), ("idle", 6), ("cross", 8), ("idle", 6), ("both", 6), ("idle", 6)]
    res = {}
    for name, models in (("규칙", []), ("학습 모델", [act_m, beh_m])):
        pa.MODE = "smooth"; pa.args.behavior = True; pa.models = models
        pa.SEQ_LEN = max([m["window"] for m in models], default=1)
        accs = []
        for i, base in enumerate(NEW_PEOPLE):
            log = []
            f, total = program(seg, base)
            run({1: f}, total, seed=900 + i, frame_log=log)
            accs.append(frame_acc(log))
        side = []
        f, total = program(rep("both"), SIDE)
        log = []; c, _, _ = run({1: f}, total, seed=950, frame_log=log)
        side = frame_acc(log).get("both"), c.get("만세", 0)
        beh = {}
        for kind in ("passing", "glance", "interested"):
            outs = []
            for r in range(3):
                f, total = walker(kind, seed=700 + r, sw=(65, 90, 105)[r], speed=(1.5, 1.8, 2.1)[r], look=2.2 - 0.2 * r)
                _, b, _ = run({1: f}, total, seed=800 + r)
                outs.append(next(iter(b), "관찰 중"))
            beh[kind] = outs
        res[name] = {"동작 프레임 정확도(낯선 사람 2명)": accs, "측면 만세(프레임 정확도, 횟수)": side, "행동": beh}
        print(f"\n### {name}\n" + json.dumps(res[name], ensure_ascii=False, indent=1))
    json.dump(res, open(f"{OUT}/evaluate{tag}.json", "w"), ensure_ascii=False, indent=1)


# ---------------- 뼈대 영상 ----------------
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]


def cmd_video():
    import av, cv2
    pa.MODE = "smooth"; pa.args.behavior = False; pa.models = []; pa.SEQ_LEN = 1
    seg = [("idle", 3), ("both", 3), ("idle", 2), ("one", 3), ("idle", 2), ("wave", 4), ("idle", 3)]
    f, total = program(seg, FRONT)
    log = []
    run({1: f}, total, seed=1, frame_log=log, tail=0)
    pa.MODE = "raw"
    log_raw = []
    run({1: f}, total, seed=1, frame_log=log_raw, tail=0)
    out = av.open(f"{OUT}/pose_demo.mp4", "w"); st = out.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = 640, 360, "yuv420p"
    for (t, _, truth, lab, _, _, k), (_, _, _, lab_raw, _, _, _) in zip(log, log_raw):
        img = np.full((H, W, 3), 30, np.uint8)
        for a, b in BONES:
            if k[a, 2] > 0.5 and k[b, 2] > 0.5:
                cv2.line(img, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), (0, 255, 0), 4)
        for x, y, c in k:
            if c > 0.5:
                cv2.circle(img, (int(x), int(y)), 6, (0, 200, 255), -1)
        cv2.putText(img, f"t={t:4.1f}s  truth={truth}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"raw   : {pa.EN.get(lab_raw, lab_raw)}", (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 255), 3)
        cv2.putText(img, f"smooth: {pa.EN.get(lab, lab)}", (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3)
        fr = av.VideoFrame.from_ndarray(cv2.resize(img, (640, 360))[:, :, ::-1].copy(), format="rgb24")
        for p in st.encode(fr):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    print(f"저장: {OUT}/pose_demo.mp4 ({len(log)}프레임, {len(log) / FPS:.0f}초)")


if __name__ == "__main__":
    {"rules": cmd_rules, "behavior": cmd_behavior, "record": cmd_record,
     "evaluate": cmd_evaluate, "video": cmd_video,
     # 2차: 사람 6명(각도·거리 다양) + arms_up 라벨로 다시 녹화·평가
     "record2": lambda: cmd_record(TRAIN_PEOPLE_V2, ACTIONS_V2, "data_v2"),
     "evaluate2": lambda: cmd_evaluate("models_v2", "_v2")}[CMD]()
