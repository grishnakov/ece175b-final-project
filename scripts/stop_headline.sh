#!/usr/bin/env bash
# Graceful early stop for the CD headline run.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${HEADLINE_CONFIG:-configs/wikiart_v2.yaml}"
NAME=$(awk '/^name:/{print $2; exit}' "$CONFIG")
[[ -z "${NAME:-}" ]] && { echo "[stop_headline] FATAL: could not parse name: from $CONFIG"; exit 1; }
RUN_DIR="experiments/$NAME"
STOP_FILE="$RUN_DIR/STOP"

mkdir -p "$RUN_DIR"
touch "$STOP_FILE"
echo "[stop_headline] created STOP marker: $STOP_FILE (supervisor will not relaunch)."

pids=$(pgrep -f "train.py --config $CONFIG" || true)
if [[ -n "$pids" ]]; then
    echo "[stop_headline] SIGTERM -> training PIDs: $(echo "$pids" | tr '\n' ' ')"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
        sleep 1
        pgrep -f "train.py --config $CONFIG" >/dev/null 2>&1 || break
    done
    leftover=$(pgrep -f "train.py --config $CONFIG" || true)
    if [[ -n "$leftover" ]]; then
        echo "[stop_headline] still alive -> SIGKILL: $(echo "$leftover" | tr '\n' ' ')"
        kill -9 $leftover 2>/dev/null || true
    fi
else
    echo "[stop_headline] no live training process found (already stopped, or supervisor between retries)."
fi

sup=$(pgrep -f "scripts/run_headline.sh" || true)
if [[ -n "$sup" ]]; then
    echo "[stop_headline] signaling supervisor to exit: $(echo "$sup" | tr '\n' ' ')"
    kill $sup 2>/dev/null || true
fi

echo "[stop_headline] done. Training stopped; all checkpoints kept."
echo "[stop_headline] resume later:  rm $STOP_FILE && tmux new -s $NAME -d 'scripts/run_headline.sh'"
