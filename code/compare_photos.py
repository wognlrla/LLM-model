"""8단계 세 모드를 '같은 사진'으로 비교한다 (실시간 앱 app.py와 같은 검출·재라벨 규칙).
  python compare_photos.py <yolo|relabel|yoloe> [사진들...]     기본 사진: media/objects/*.jpg
결과: 박스마다 검출 라벨·신뢰도·재라벨 결과를 출력하고, 박스를 그린 사진을 out_compare/<모드>_<이름>.jpg 로 저장.
실시간으로 같은 물체를 세 번 보여 주면 위치·조명이 달라지지만, 이 방식은 세 모드가 완전히 같은 입력을 받는다.
"""
import glob, json, os, sys, time
import cv2, numpy as np, torch

MODE = sys.argv[1]
PHOTOS = sys.argv[2:] or sorted(glob.glob("media/objects/*.jpg")) + sorted(glob.glob("media/objects/crops/*.jpg"))
# media/objects/crops/ 의 사진은 실시간 yolo 앱이 이미 잘라 둔 물체다.
# relabel 모드에서는 app.py와 똑같이 "크롭 전체"를 재라벨한다 (YOLO 원래 라벨은 아래 표).
CROP_YOLO = {"laptop_closed": "laptop", "laptop_open": "laptop", "keyboard": "keyboard",
             "monitor_front": "tv", "monitor_back": "tv"}
W = 1280                                   # app.py와 같은 검출 해상도
CONF = {"yolo": 0.5, "relabel": 0.35, "yoloe": 0.25}[MODE]
MIN_SIDE, DIM, LABEL_MIN = 48, 512, 0.20
LABELS = [
    ("drum washing machine", "드럼 세탁기"), ("top-load washing machine", "통돌이 세탁기"),
    ("oven", "오븐"), ("microwave", "전자레인지"), ("refrigerator", "냉장고"),
    ("cell phone", "휴대폰"), ("battery pack", "원통형 전지 포장"), ("AA battery", "건전지"),
    ("laptop", "노트북"), ("monitor", "모니터"), ("keyboard", "키보드"), ("mouse", "마우스"),
    ("cup", "컵"), ("water bottle", "물병"), ("book", "책"), ("backpack", "가방"), ("chair", "의자"),
    ("product box", "제품 상자"), ("cardboard box", "종이 상자"),          # 실습 추가 라벨
]
KO = dict(LABELS)
os.makedirs("out_compare", exist_ok=True)

if MODE == "yoloe":
    from ultralytics import YOLOE
    names = [en for en, _ in LABELS]
    det = YOLOE("yoloe-11s-seg.pt")
    det.set_classes(names, det.get_text_pe(names))
else:
    from ultralytics import YOLO
    det = YOLO("yolo11n.pt")

emb = None
if MODE == "relabel":                       # 재라벨에만 임베딩 모델이 필요하다
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("google/embeddinggemma-2", device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
    with torch.inference_mode():
        label_vecs = np.asarray(emb.encode([f"a photo of a {en}" for en, _ in LABELS],
            prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True), dtype=np.float32)

def relabel(rgb):
    v = emb.encode([{"image": rgb}], truncate_dim=DIM, normalize_embeddings=True)
    sim = np.asarray(v, dtype=np.float32)[0] @ label_vecs.T
    top = np.argsort(-sim)[:3]
    return int(top[0]), float(sim[top[0]]), [(LABELS[k][1], round(float(sim[k]), 3)) for k in top]

results = []
for path in PHOTOS:
    stem = os.path.splitext(os.path.basename(path))[0]
    if MODE == "relabel" and "/crops/" in path:          # 크롭 전체를 바로 재라벨
        from PIL import Image
        j, s, top3 = relabel(Image.open(path).convert("RGB"))
        name = CROP_YOLO.get(stem, stem)
        results.append({"photo": "crops/" + os.path.basename(path), "det_ms": 0, "boxes": [{
            "det": name + " (실시간 yolo)", "conf": 1.0, "label": LABELS[j][1] if s >= LABEL_MIN else name,
            "score": round(s, 3), "top3": top3}]})
        continue
    img = cv2.imread(path)
    img = cv2.resize(img, (W, int(img.shape[0] * W / img.shape[1])), interpolation=cv2.INTER_AREA)
    t = time.perf_counter()
    r = det.predict(img, conf=CONF, verbose=False)[0]
    det_ms = (time.perf_counter() - t) * 1000
    boxes = []
    for (x1, y1, x2, y2), c, cf in zip(r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        name = det.names[c]
        b = {"det": name, "conf": round(cf, 2), "box": [x1, y1, x2, y2], "label": KO.get(name, name)}
        if emb is not None and name != "person" and min(x2 - x1, y2 - y1) >= MIN_SIDE:
            from PIL import Image
            j, s, top3 = relabel(Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
            b["label"] = LABELS[j][1] if s >= LABEL_MIN else name
            b["score"], b["top3"] = round(s, 3), top3
        boxes.append(b)
        color = (0, 200, 255) if MODE == "relabel" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{name} {cf:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    out = f"out_compare/{MODE}_{os.path.basename(path)}"
    cv2.imwrite(out, img)
    label = ("crops/" if "/crops/" in path else "") + os.path.basename(path)
    results.append({"photo": label, "det_ms": round(det_ms), "boxes": boxes})

for r in results:
    print(f"## {r['photo']}  (검출 {r['det_ms']} ms)")
    if not r["boxes"]:
        print("   박스 없음")
    for b in r["boxes"]:
        extra = f" → 재라벨 {b['label']} ({b['score']}) | 후보 {b['top3']}" if "score" in b else ""
        print(f"   {b['det']:22s} {b['conf']:.2f}{extra}")
json.dump(results, open(f"out_compare/{MODE}.json", "w"), ensure_ascii=False, indent=1)
