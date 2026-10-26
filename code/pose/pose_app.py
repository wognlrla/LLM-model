import os, time, threading, collections, argparse, csv
import cv2, numpy as np, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# [보드 대응] POSE_OFFLINE=1 이면 카메라·YOLO를 열지 않고 판정 로직(step)만 쓴다.
# pose_sim.py가 합성 관절 데이터를 이 로직에 그대로 넣어 실험할 때 사용한다.
OFFLINE = os.environ.get("POSE_OFFLINE") == "1"

# ---------------- 실행 옵션 ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
# [보드 대응] USB 웹캠 대신 VIGI IP 카메라(RTSP). stream1=2560x1440, stream2=848x480, 숫자면 USB 웹캠
ap.add_argument("--source", default="stream1", help="stream1 | stream2 | 0(USB 웹캠)")
ap.add_argument("--flip", action="store_true", help="카메라가 거꾸로 달렸을 때 180도 회전")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=8000)
args, _ = ap.parse_known_args()
MODE = "raw" if args.raw else "smooth"

W, H = 1280, 720
SMOOTH_S = 0.6        # 다수결 창(초)
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정 창(초)
FACE_TH = 0.6         # 정면으로 볼 facing 점수
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초)
ML_TH = 0.7           # 모델 확률 기준
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "..."}
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}

if OFFLINE:
    det = None
else:
    from ultralytics import YOLO
    det = YOLO("yolo11n-pose.pt")         # 첫 실행 시 자동 다운로드
models = [load_model(p) for p in args.model]
SEQ_LEN = max([m["window"] for m in models], default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} "
      f"models={[(m['task'], m['labels']) for m in models]}")

frame_lock, ev_lock = threading.Lock(), threading.Lock()
latest_jpeg = None
tracks = {}
pairs = {}            # (tid_a, tid_b) -> deque[(t, bool)]
pair_on = set()
events = collections.deque(maxlen=50)
counts = collections.Counter()
beh_counts = collections.Counter()   # 사람(추적 ID) 단위 최종 행동
stats = {"fps": 0.0, "people": 0}
rec = None
if args.record:
    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    rec_f = open(args.record, "a", newline="")
    rec = csv.writer(rec_f)

def new_track(now):
    return {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
            "beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=SEQ_LEN),
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def is_wave(hist):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 2번 이상 방향을 바꾸면 손 흔들기"""
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.15:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2

def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴 판정"""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝에서부터 정면 연속 시간
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def log_event(now, who, label):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": label})
        counts[label] += 1

def close_track(tid, st, now):
    """추적이 끝난 사람의 요약을 남긴다"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1

def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            sw = (ba[1] + bb[1]) / 2
            best = 1e9
            for wa in (LW, RW):
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                            best = min(best, np.linalg.norm(a[wa, :2] - b[wb, :2]))
            key = tuple(sorted((ids[i], ids[j])))
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, best < 0.5 * sw))
            found.add(key)
    return found

def update_person(tid, k, now):
    st = tracks.setdefault(tid, new_track(now))
    dt = now - st["seen"]
    st["seen"] = now
    b = body(k)
    if b is None:
        return st
    raw = pose_label(k, b)
    fa = facing_score(k)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt
    st["n"] += 1
    if raw in ("한팔들기", "만세"):           # 손 흔들기용: 올라간 손목의 x (어깨 중심 기준, 어깨 너비 단위)
        w = LW if k[LW, 1] < k[RW, 1] else RW
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec:
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        if label == "한팔들기" and is_wave(recent):
            label = "손흔들기"

    # ---- 학습 모델 (5프레임마다) ----
    if models and st["n"] % 5 == 0:
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:
                lab, p = predict(m, seq[-m["window"]:])
                st["ml"][m["task"]] = (lab, p) if p >= ML_TH else None
    ml_act = st["ml"].get("action")
    if ml_act:
        label = "기본" if ml_act[0] == "idle" else ml_act[0]

    if label != st["label"]:
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
            st["best"] = bs
    return st


def step(ids, kps, boxes, now, vis=None):
    """[보드 대응] 한 프레임의 판정 (원래 capture_loop 안에 있던 코드를 그대로 함수로 뺌).
    카메라 영상(capture_loop)과 합성 관절 데이터(pose_sim.py)가 같은 코드를 쓴다."""
    stats["people"] = len(ids)
    for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
        st = update_person(tid, k, now)
        if vis is not None:
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---- 2인 동작 (악수) ----
    active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
    for key in list(pairs):
        if key not in active:
            pairs.pop(key); pair_on.discard(key); continue
        if MODE == "raw":
            on = pairs[key][-1][1]
        else:
            v, frac = majority(pairs[key], now, SMOOTH_S)
            on = bool(v) and frac >= 0.6
        if on and key not in pair_on:
            log_event(now, f"#{key[0]}+#{key[1]}", "악수")
        (pair_on.add if on else pair_on.discard)(key)
    if vis is not None:
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

    # 3초 이상 안 보인 사람은 추적 종료 + 요약
    for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
        close_track(tid, tracks.pop(tid), now)


class LatestFrame:
    """[보드 대응] RTSP는 검출보다 빨리 프레임이 들어오므로 최신 프레임만 남긴다 (8단계 app.py와 같음)"""
    def __init__(self, source):
        if source.isdigit():                                   # USB 웹캠 (안내문 원래 방식)
            cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        else:                                                  # VIGI 카메라 RTSP
            url = open(os.path.expanduser("~/eg2-lab/.camera_url")).read().strip()
            url = url.rsplit("/", 1)[0] + "/" + source
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        assert cap.isOpened(), "카메라를 열 수 없음 (.camera_url 또는 연결 확인)"
        self.cap, self.frame, self.lock = cap, None, threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.1); continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            f, self.frame = self.frame, None
        if f is None:
            return False, None
        if f.shape[1] != W:
            f = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1])), interpolation=cv2.INTER_AREA)
        if args.flip:
            f = cv2.rotate(f, cv2.ROTATE_180)
        return True, f


def capture_loop():
    global latest_jpeg
    cap = LatestFrame(str(args.cam) if args.source.isdigit() else args.source)
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        r = det.track(frame, persist=True, conf=args.conf, half=True, verbose=False)[0]
        now = time.time()
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()
        step(ids, kps, boxes, now, vis)

        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

app = FastAPI()

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in tracks.items()}}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명`;
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
