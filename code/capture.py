"""IP 카메라(RTSP)에서 10초 녹화 + 2초마다 프레임 저장.

보드의 NVIDIA ffmpeg에는 RTSP 입력과 H.264 디코더가 없어서 PyAV로 처리한다.
카메라 주소(계정·비밀번호 포함)는 화면에 찍히지 않도록 ~/eg2-lab/.camera_url 파일에서 읽는다.
  사용법: python capture.py [초=10] [채널=stream1|stream2]
"""
import os, sys, time, av
from eg2 import extract_frames

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
url_file = os.path.expanduser("~/eg2-lab/.camera_url")
url = open(url_file).read().strip()
if len(sys.argv) > 2:                      # 채널 바꾸기 (stream1=고화질, stream2=저화질)
    url = url.rsplit("/", 1)[0] + "/" + sys.argv[2]
masked = url.split("@")[-1]                # 계정·비밀번호 가리기

os.makedirs("media", exist_ok=True)
print(f"연결: rtsp://***@{masked}")
src = av.open(url, options={"rtsp_transport": "tcp", "stimeout": "5000000", "timeout": "5000000"})
vin = src.streams.video[0]
print(f"코덱 {vin.codec_context.name}, {vin.codec_context.width}x{vin.codec_context.height}, {float(vin.average_rate or 0):.0f}fps")

dst = av.open("media/clip.mp4", "w")
vout = dst.add_stream_from_template(vin)    # 다시 인코딩하지 않고 그대로 저장(-c:v copy와 같음)
start = None; n = 0; t0 = time.time()
for pkt in src.demux(vin):
    if pkt.dts is None:
        continue
    if start is None:
        if not pkt.is_keyframe:            # 키프레임부터 저장해야 재생된다
            continue
        start = pkt.pts
    if (pkt.pts - start) * pkt.time_base >= secs or time.time() - t0 > secs + 30:
        break
    pkt.pts -= start; pkt.dts -= start; pkt.stream = vout
    dst.mux(pkt); n += 1
dst.close(); src.close()
print(f"녹화 완료: media/clip.mp4 ({n} 패킷, {os.path.getsize("media/clip.mp4")/1e6:.1f} MB)")
print("프레임:", extract_frames("media/clip.mp4"))
