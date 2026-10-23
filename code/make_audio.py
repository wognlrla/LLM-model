"""5단계 샘플 만들기 (보드 ffmpeg에 lavfi가 없어 파이썬으로 대체)
  python make_audio.py            → beep.wav, noise.wav 생성 + raw_*.wav 를 16kHz 모노로 변환
  python make_audio.py mic 5      → 카메라 마이크로 5초 녹음(raw_mic.wav) 후 변환
"""
import glob, os, sys
import numpy as np, soundfile as sf, librosa

SR = 16000
os.makedirs("media", exist_ok=True)

def to16k(src):
    y, sr = librosa.load(src, sr=SR, mono=True)          # 리샘플 + 모노
    dst = src.replace("raw_", "")
    sf.write(dst, y, SR, subtype="PCM_16"); return dst, len(y) / SR

if len(sys.argv) > 1 and sys.argv[1] == "mic":           # 카메라(VIGI) 내장 마이크 녹음
    import av
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    url = open(os.path.expanduser("~/eg2-lab/.camera_url")).read().strip()
    chunks = []; sr = None
    with av.open(url, options={"rtsp_transport": "tcp", "timeout": "5000000"}) as c:
        a = c.streams.audio[0]
        for fr in c.decode(a):
            x = fr.to_ndarray().astype(np.float32).reshape(-1)
            if x.dtype != np.float32 or np.abs(x).max() > 1.5: x = x / 32768.0
            chunks.append(x); sr = fr.sample_rate
            if sum(len(k) for k in chunks) >= secs * sr: break
    sf.write("media/raw_mic.wav", np.concatenate(chunks), sr, subtype="PCM_16")
    print(f"카메라 마이크 녹음: {secs:.0f}초, 원본 {sr}Hz")
else:
    t = np.arange(int(3 * SR)) / SR                         # 1 kHz 경보음 3초
    sf.write("media/raw_beep.wav", 0.5 * np.sin(2 * np.pi * 1000 * t), SR, subtype="PCM_16")
    rng = np.random.default_rng(0)                          # 핑크 노이즈 5초
    w = rng.standard_normal(5 * SR); f = np.fft.rfftfreq(len(w), 1 / SR); f[0] = 1
    pink = np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), len(w)); pink = 0.3 * pink / np.abs(pink).max()
    sf.write("media/raw_noise.wav", pink, SR, subtype="PCM_16")

for src in sorted(glob.glob("media/raw_*.wav")):
    dst, dur = to16k(src)
    info = sf.info(dst)
    print(f"{dst}: {info.samplerate}Hz, {info.channels}채널, {dur:.1f}초")
