import time
from eg2 import load

model = load()  # 텍스트 전용

docs = [
    "환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
    "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
    "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
    "Jetson Orin Nano는 67 TOPS의 AI 성능을 제공한다.",
]
queries = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]

d = model.encode(docs, prompt_name="Document", normalize_embeddings=True)

for dim in [768, 256, 128]:
    t = time.perf_counter()
    q = model.encode(queries, prompt_name="SearchQuery",
                     truncate_dim=dim, normalize_embeddings=True)
    ms = (time.perf_counter() - t) * 1000
    dd = d[:, :dim]
    dd = dd / (dd ** 2).sum(1, keepdims=True) ** 0.5   # 자른 뒤 재정규화
    sim = q @ dd.T
    print(f"\n[dim={dim}] {ms:.0f} ms")
    for qi, row in zip(queries, sim):
        best = row.argmax()
        print(f"  {qi} -> {docs[best][:25]}... ({row[best]:.3f})")
