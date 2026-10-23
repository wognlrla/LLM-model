import glob, time
import soundfile as sf
from eg2 import load

model = load(audio=True)

files = sorted(f for f in glob.glob("media/*.wav") if "/raw_" not in f)
for f in files:
    info = sf.info(f)
    assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님"

t = time.perf_counter()
aud_emb = model.encode([{"audio": f} for f in files], normalize_embeddings=True)
print(f"오디오 {len(files)}개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["덥다고 냉방을 요청하는 목소리", "날카로운 경보음", "웅웅거리는 배경 소음"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

sim = q @ aud_emb.T
for qi, row in zip(queries, sim):
    best = row.argmax()
    print(f"{qi} -> {files[best].split("/")[-1]} ({row[best]:.3f})")
