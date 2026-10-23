import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"

def load(vision=False, audio=False):
    cfg = {}
    if not vision: cfg["vision_config"] = None   # 미사용 인코더는 로드하지 않음
    if not audio:  cfg["audio_config"] = None
    return SentenceTransformer(
        MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},  # float16 금지
        config_kwargs=cfg,
    )


# --- Jetson 보드 대응 (2026-10-23 추가) ---
# 보드의 NVIDIA ffmpeg 빌드에는 소프트웨어 H.264 디코더·libx264가 없고 torchcodec도 로드되지 않는다.
# 그래서 동영상은 PyAV(자체 ffmpeg 내장)로 초당 1프레임씩 풀어 배열로 넘긴다.
def video_frames(path, fps=1.0):
    """동영상을 fps 간격으로 디코딩해 (프레임수, 높이, 너비, 3) uint8 배열로 돌려준다."""
    import av, numpy as np
    with av.open(path) as c:
        st = c.streams.video[0]
        step = max(1, round(float(st.average_rate or 1) / fps))
        frames = [f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(st)) if i % step == 0]
    return np.stack(frames)


def extract_frames(path, out_pattern="media/frame_%02d.jpg", every_sec=2.0):
    """ffmpeg -vf fps=0.5 대신: every_sec초마다 프레임을 JPG로 저장하고 파일 목록을 돌려준다."""
    from PIL import Image
    saved = []
    for i, fr in enumerate(video_frames(path, fps=1.0 / every_sec), start=1):
        p = out_pattern % i
        Image.fromarray(fr).save(p, quality=90)
        saved.append(p)
    return saved
