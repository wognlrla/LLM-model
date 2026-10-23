#!/bin/bash
# 사용법: ./guard_run.sh <명령...>  — 보드 사용 가능 메모리가 400MB 아래로 떨어지면 그 명령을 강제 종료
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
  if [ "$a" -lt 400 ]; then kill -9 $pid; echo "[보호] 메모리 부족(${a}MB)으로 중단"; fi
  sleep 1
done
wait $pid 2>/dev/null
