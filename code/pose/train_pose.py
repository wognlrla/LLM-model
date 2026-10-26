"""9-6: --record로 모은 CSV로 GRU 동작/행동 분류 모델 학습
파일 이름(확장자 제외)이 정답 라벨이다. 예) data/action/wave.csv → "wave"
사용법:
  python train_pose.py --data data/action --task action   --window 30 --out models/action.pt
  python train_pose.py --data data/behavior --task behavior --window 80 --stride 10 --out models/behavior.pt
"""
import argparse, csv, glob, os, collections
import numpy as np, torch, torch.nn as nn
from pose_model import PoseGRU, N_FEAT

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--task", choices=["action", "behavior"], required=True)
ap.add_argument("--window", type=int, default=30, help="시퀀스 길이(프레임)")
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--val", type=float, default=0.2, help="각 시퀀스 뒷부분을 검증용으로 사용")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load_sequences(path):
    """추적 ID별로 시간순 정렬, 0.5초 이상 끊기면 다른 시퀀스로 나눈다"""
    by_tid = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 5 + 51:
                continue
            ts, tid = float(row[0]), row[1]
            fa, spd = float(row[3]), float(row[4])
            feat = np.array([float(x) for x in row[5:]] + [fa, spd], dtype=np.float32)
            by_tid[tid].append((ts, feat))
    seqs = []
    for rows in by_tid.values():
        rows.sort(key=lambda r: r[0])
        cur = [rows[0][1]]
        for (t0, _), (t1, f) in zip(rows, rows[1:]):
            if t1 - t0 > 0.5:
                seqs.append(cur); cur = []
            cur.append(f)
        seqs.append(cur)
    return [np.stack(s) for s in seqs if len(s) >= args.window]

def windows(seq):
    return [seq[i:i + args.window] for i in range(0, len(seq) - args.window + 1, args.stride)]

files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
labels = [os.path.splitext(os.path.basename(p))[0] for p in files]
assert len(labels) >= 2, "라벨(CSV 파일)이 2개 이상 필요"
Xtr, ytr, Xva, yva = [], [], [], []
for ci, p in enumerate(files):
    for seq in load_sequences(p):
        cut = int(len(seq) * (1 - args.val))      # 시간순 분할: 앞 80% 학습, 뒤 20% 검증 (겹침 방지)
        tr, va = windows(seq[:cut]), windows(seq[cut:])
        Xtr += tr; ytr += [ci] * len(tr); Xva += va; yva += [ci] * len(va)
    print(f"{labels[ci]:>15}: 학습 {ytr.count(ci)}, 검증 {yva.count(ci)} 윈도")
assert Xtr and Xva, "윈도가 부족함: 녹화 시간을 늘리거나 --window를 줄일 것"

Xtr, Xva = torch.tensor(np.stack(Xtr)), torch.tensor(np.stack(Xva))
ytr, yva = torch.tensor(ytr), torch.tensor(yva)
mean = Xtr.reshape(-1, N_FEAT).mean(0)
std = Xtr.reshape(-1, N_FEAT).std(0) + 1e-6
norm = lambda x: ((x - mean) / std).to(dev)

model = PoseGRU(N_FEAT, len(labels), args.hidden).to(dev)
w = 1.0 / torch.bincount(ytr, minlength=len(labels)).float().clamp(min=1)
loss_fn = nn.CrossEntropyLoss(weight=(w / w.sum() * len(labels)).to(dev))   # 클래스 불균형 보정
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best, best_state = -1, None
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        x = norm(Xtr[idx]) + 0.05 * torch.randn(len(idx), args.window, N_FEAT, device=dev)  # 약한 잡음 증강
        loss = loss_fn(model(x), ytr[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(norm(Xva)).argmax(-1).cpu()
    acc = (pred == yva).float().mean().item()
    if acc > best:
        best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

model.load_state_dict(best_state)
with torch.no_grad():
    pred = model(norm(Xva)).argmax(-1).cpu()
cm = np.zeros((len(labels), len(labels)), dtype=int)
for t, p in zip(yva.tolist(), pred.tolist()):
    cm[t, p] += 1
print(f"\n최고 검증 정확도 {best:.3f}\n혼동 행렬 (행=정답, 열=예측)")
print(" " * 16 + "".join(f"{l[:8]:>9}" for l in labels))
for l, row in zip(labels, cm):
    print(f"{l:>15} " + "".join(f"{v:>9}" for v in row))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"state": best_state, "labels": labels, "task": args.task, "window": args.window,
            "hidden": args.hidden, "mean": mean.tolist(), "std": std.tolist()}, args.out)
print(f"\n저장: {args.out}  (task={args.task}, window={args.window}, labels={labels})")
