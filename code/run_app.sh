#!/bin/bash
# 사용법: ./run_app.sh <모드> [추가 인자...]   예) ./run_app.sh relabel --flip
# 이미 떠 있는 app.py를 끄고, 지정한 모드로 백그라운드 실행한다. 로그: app_<모드>.log
cd ~/eg2-lab
for p in $(pgrep -u "$USER" -f "^/home/$USER/eg2/bin/python app\.py"); do kill -9 "$p"; done
sleep 2
mode=$1; shift
nohup ~/eg2/bin/python app.py --mode "$mode" "$@" > "app_${mode}.log" 2>&1 < /dev/null &
echo "실행: $mode $* (pid $!)"
