#!/bin/bash
# 보드 사용 가능 메모리가 400MB 아래로 내려가면 우리 app.py만 끈다 (상시 서비스 보호)
while pgrep -u "$USER" -f "^/home/$USER/eg2/bin/python app\.py" >/dev/null; do
  avail=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
  if [ "$avail" -lt 400 ]; then
    pkill -9 -u "$USER" -f "^/home/$USER/eg2/bin/python app\.py"
    echo "$(date +%T) 메모리 부족(${avail}MB)으로 app.py 종료" >> ~/eg2-lab/mem_guard.log
  fi
  sleep 2
done
