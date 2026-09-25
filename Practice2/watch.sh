#!/usr/bin/env bash
# PRACTICE 2 — watch.sh : giám sát tự động hàng đợi train
# Chạy:  nohup bash practice2/watch.sh >/dev/null 2>&1 &
# Xem:   cat practice2/logs/STATUS.txt          (b bức tranh hiện tại)
#        tail -f practice2/logs/progress.log    (vết theo thời gian)
set -u
cd "$(dirname "$0")/.."

MASTER="practice2/logs/master.log"
STATUS="practice2/logs/STATUS.txt"
PROGRESS="practice2/logs/progress.log"
INTERVAL=60
TOTAL=$(grep -c "^run --model" practice2/run_experiments.sh 2>/dev/null || true)
[ "${TOTAL:-0}" -ge 1 ] || TOTAL=7
last_line=""

while true; do
    ts=$(date "+%H:%M:%S")
    alive=$(pgrep -f "run_experiments.sh" >/dev/null && echo "DANG CHAY" || echo "DA DUNG")

    done_n=$(grep -c "===== DONE" "$MASTER" 2>/dev/null || true)
    cur=$(grep "^===== \[" "$MASTER" 2>/dev/null | grep START | tail -1 | sed -E 's/.*--run-name ([^ ]+).*/\1/')
    ep=$(grep -E "^  Epoch [0-9]+/" "$MASTER" 2>/dev/null | tail -1)
    all_done=$(grep -c "TẤT CẢ EXPERIMENTS XONG" "$MASTER" 2>/dev/null || true)

    # epoch hiện tại của run đang chạy (đếm từ lần START cuối)
    ep_now=$(awk '/START/{n=0} /  Epoch /{n++} END{print n+0}' "$MASTER" 2>/dev/null)

    {
        echo "===== PRACTICE 2 — trang thái $ts $(date +%d/%m/%Y) ====="
        echo "Hang doi : $alive | xong $done_n/$TOTAL experiments"
        echo "Dang chay: ${cur:-<chưa bat dau>}"
        echo "Epoch gan nhat: $ep_now/6 — ${ep:-<chua xong epoch nao, ~10 ph/epoch>}"
        if [ "${all_done:-0}" -gt 0 ]; then
            echo ">>> TOT CAT DA XONG <<< Tong hop ket qua: bash practice2/collect_results.sh"
        fi
        # bảng tổng hợp các run đã xong (từ log từng run nếu có)
        echo "--- Ket qua tam tinh (tu cac Epoch cuoi trong master.log) ---"
        awk '/--run-name/{split($0,a,"--run-name "); r=a[2]; sub(/ .*/,"",r)}
             /^  Epoch /{last[r]=$0}
             END{for(k in last) printf "  %-28s %s\n", k, last[k]}' "$MASTER" 2>/dev/null | sort
    } > "$STATUS"

    # chi ghi progress.log khi có dòng epoch mới (chống rác log)
    if [ "$ep" != "$last_line" ] && [ -n "$ep" ]; then
        echo "[$ts] $cur: $ep" >> "$PROGRESS"
        last_line="$ep"
    fi

    [ "${all_done:-0}" -gt 0 ] && { echo "[$ts] ALL DONE" >> "$PROGRESS"; break; }
    if [ "$alive" = "DA DUNG" ] && [ "${all_done:-0}" -eq 0 ]; then
        echo "[$ts] CANH BAO: hang doi dung ma chua co ALL DONE — kiem tra master.log!" >> "$PROGRESS"
    fi

    sleep $INTERVAL
done
