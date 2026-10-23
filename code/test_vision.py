import glob, time
from eg2 import load, video_frames

model = load(vision=True)

images = sorted(glob.glob("media/*.jpg"))
t = time.perf_counter()
img_emb = model.encode([{"image": p} for p in images], normalize_embeddings=True)
print(f"이미지 {len(images)}장: {(time.perf_counter()-t)*1000:.0f} ms")

t = time.perf_counter()
vid_emb = model.encode({"video": video_frames("media/clip.mp4")}, normalize_embeddings=True)
print(f"동영상 1개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

sim = q @ img_emb.T
for qi, row in zip(queries, sim):
    order = row.argsort()[::-1][:3]
    print(qi, "->", [(images[i].split("/")[-1], round(float(row[i]), 3)) for i in order])

print("질의-동영상 유사도:", (q @ vid_emb).round(3))
