"""6단계 성능 측정:  python bench.py <구성>
구성: text | vision_img | vision_vid | audio | full
- 로드 후 RAM: 보드 전체 사용량(MemTotal-MemAvailable, tegrastats RAM과 같은 기준)과 늘어난 양
- 지연: 같은 입력을 2번 encode, 두 번째 값을 표에 쓴다
- GPU 사용률: 두 번째 실행 동안 /sys/devices/platform/gpu.0/load 를 50ms마다 읽은 평균·최대
"""
import glob, json, sys, threading, time
import torch
from eg2 import load, video_frames

cfg = sys.argv[1]

def used_mb():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return (m["MemTotal"] - m["MemAvailable"]) // 1024

class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.gpu, s.ram, s.on = [], [], True
    def run(s):
        while s.on:
            s.gpu.append(int(open("/sys/devices/platform/gpu.0/load").read()) / 10)
            s.ram.append(used_mb()); time.sleep(0.05)

QUERIES = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습"]
IMAGES = sorted(glob.glob("media/scene_*.jpg"))[:5]
WAVS = ["media/speech.wav", "media/beep.wav", "media/noise.wav"]

base = used_mb()
t = time.perf_counter()
model = load(vision=cfg in ("vision_img", "vision_vid", "full"), audio=cfg in ("audio", "full"))
torch.cuda.synchronize(); load_s = time.perf_counter() - t
after = used_mb()
params = sum(p.numel() for p in model.parameters()) / 1e6

vid = video_frames("media/clip.mp4")[:10] if cfg in ("vision_vid", "full") else None   # 10초 = 1fps 10프레임

def job():
    if cfg == "text":
        model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
    elif cfg == "vision_img":
        model.encode([{"image": p} for p in IMAGES], normalize_embeddings=True)
    elif cfg == "vision_vid":
        model.encode({"video": vid}, normalize_embeddings=True)
    elif cfg == "audio":
        model.encode([{"audio": f} for f in WAVS], normalize_embeddings=True)
    elif cfg == "full":   # 혼합: 질의 2 + 이미지 5 + 동영상 10초 + WAV 3
        model.encode(QUERIES, prompt_name="SearchQuery", normalize_embeddings=True)
        model.encode([{"image": p} for p in IMAGES], normalize_embeddings=True)
        model.encode({"video": vid}, normalize_embeddings=True)
        model.encode([{"audio": f} for f in WAVS], normalize_embeddings=True)
    torch.cuda.synchronize()

t = time.perf_counter(); job(); run1 = (time.perf_counter() - t) * 1000
s = Sampler(); s.start()
t = time.perf_counter(); job(); run2 = (time.perf_counter() - t) * 1000
s.on = False; s.join()

r = dict(구성=cfg, 파라미터M=round(params), 로드초=round(load_s, 1), 로드전RAM=base, 로드후RAM=after,
         증가RAM=after - base, 실행중최대RAM=max(s.ram), 지연1차ms=round(run1), 지연2차ms=round(run2),
         GPU평균=round(sum(s.gpu) / len(s.gpu)), GPU최대=round(max(s.gpu)))
print("RESULT", json.dumps(r, ensure_ascii=False))
