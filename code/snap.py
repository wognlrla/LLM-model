"""카메라에서 사진 한 장 찍기:  python snap.py 이름 [flip]
   예) python snap.py 빈강의실      → media/빈강의실.jpg
       python snap.py 화이트보드 flip → 위아래 뒤집어서 저장(카메라가 거꾸로 놓였을 때)"""
import os, sys, av
from PIL import Image
name = sys.argv[1] if len(sys.argv) > 1 else "snap"
flip = len(sys.argv) > 2 and sys.argv[2] == "flip"
url = open(os.path.expanduser("~/eg2-lab/.camera_url")).read().strip()
with av.open(url, options={"rtsp_transport": "tcp", "timeout": "5000000"}) as c:
    st = c.streams.video[0]
    for f in c.decode(st):
        if f.key_frame:
            img = f.to_image(); break
if flip:
    img = img.rotate(180)
os.makedirs("media", exist_ok=True)
p = f"media/{name}.jpg"; img.save(p, quality=90); print("저장:", p, img.size)
