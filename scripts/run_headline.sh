#!/usr/bin/env bash
# CD headline run — supervised auto-resume wrapper.
# `tmux new -s wikiart_v2 -d 'scripts/run_headline.sh'`
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

CONFIG="${HEADLINE_CONFIG:-configs/wikiart_v2.yaml}"
TOTAL_STEPS="${HEADLINE_TOTAL_STEPS:-90000}"
BACKOFF="${HEADLINE_BACKOFF:-30}"

NAME=$(awk '/^name:/{print $2; exit}' "$CONFIG")
if [[ -z "${NAME:-}" ]]; then
    printf 'FATAL: could not parse name: from %s\n' "$CONFIG"; exit 1
fi
RUN_DIR="experiments/$NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
STDOUT_LOG="$RUN_DIR/run_headline_stdout.log"
SUP_LOG="$RUN_DIR/supervisor.log"
STOP_FILE="$RUN_DIR/STOP"

MAX_TRIES=200
FASTFAIL_SECS=120
MAX_NOPROGRESS=3

mkdir -p "$CKPT_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    printf '%s\n' "$msg" | tee -a "$SUP_LOG"
}

latest_ckpt() {
    ls -1 "$CKPT_DIR"/step_*.pt 2>/dev/null | sort | tail -1
}

step_of() {
    local c="${1:-}"
    [[ -z "$c" ]] && { printf '0'; return; }
    local b
    b=$(basename "$c" .pt)
    b=${b#step_}
    [[ "$b" =~ ^[0-9]+$ ]] || { printf '0'; return; }
    printf '%d' "$((10#$b))"
}

on_signal() {
    touch "$STOP_FILE" 2>/dev/null || true
    log "signal received -> STOP set; terminating training and exiting (no relaunch)."
    pkill -f "train.py --config $CONFIG" 2>/dev/null || true
    exit 0
}
trap on_signal SIGINT SIGTERM SIGHUP

CFG_STEPS=$(grep -E '^[[:space:]]+steps:' "$CONFIG" | grep -oE '[0-9]+' | head -1)
if [[ -z "$CFG_STEPS" ]]; then
    log "FATAL: could not parse training.steps from $CONFIG"
    exit 1
fi
if [[ "$CFG_STEPS" != "$TOTAL_STEPS" ]]; then
    log "FATAL: TOTAL_STEPS=$TOTAL_STEPS != config steps=$CFG_STEPS. Set them equal before launch."
    exit 1
fi

if [[ -f "$STOP_FILE" ]]; then
    log "STOP file present at startup ($STOP_FILE). Refusing to launch."
    log "  rm it to (re)start:  rm $STOP_FILE && tmux new -s $NAME -d 'scripts/run_headline.sh'"
    exit 0
fi

log "==== headline supervisor start: name=$NAME target=$TOTAL_STEPS steps, config=$CONFIG ===="

tries=0
noprog=0
while true; do
    if [[ -f "$STOP_FILE" ]]; then
        log "STOP requested ($STOP_FILE) -> graceful stop, not relaunching. rm it to resume."
        exit 0
    fi

    tries=$((tries + 1))
    if [[ $tries -gt $MAX_TRIES ]]; then
        log "ABORT: exceeded MAX_TRIES=$MAX_TRIES launch attempts."
        exit 1
    fi

    latest=$(latest_ckpt)
    cur_step=$(step_of "$latest")

    if [[ $cur_step -ge $TOTAL_STEPS ]]; then
        log "DONE: latest checkpoint at step $cur_step >= $TOTAL_STEPS. Nothing to do."
        exit 0
    fi

    resume_arg=()
    if [[ -n "$latest" ]]; then
        resume_arg=(--resume "$latest")
        log "launch #$tries: resuming from step $cur_step ($latest)"
    else
        log "launch #$tries: starting fresh from step 0"
    fi

    t0=$SECONDS
    uv run python -u train.py --config "$CONFIG" "${resume_arg[@]}" 2>&1 | tee -a "$STDOUT_LOG"
    dur=$((SECONDS - t0))

    if [[ -f "$STOP_FILE" ]]; then
        log "STOP requested during segment -> exiting cleanly without relaunch."
        log "  latest checkpoint preserved; rm $STOP_FILE to resume."
        exit 0
    fi

    new_latest=$(latest_ckpt)
    new_step=$(step_of "$new_latest")

    if [[ $new_step -ge $TOTAL_STEPS ]]; then
        log "DONE: training reached step $new_step in ${dur}s. Run complete."
        exit 0
    fi

    if [[ $new_step -gt $cur_step ]]; then
        noprog=0
        log "died at step $new_step after ${dur}s (advanced from $cur_step). Backing off ${BACKOFF}s, relaunching."
    else
        noprog=$((noprog + 1))
        if [[ $dur -lt $FASTFAIL_SECS && -n "$latest" && "$new_latest" == "$latest" ]]; then
            log "no progress: fast-fail (${dur}s) resuming from $latest -> quarantining as .corrupt"
            mv "$latest" "$latest.corrupt" 2>/dev/null || true
        else
            log "no progress this attempt (step still $new_step, ran ${dur}s). Will retry same checkpoint."
        fi
        if [[ $noprog -ge $MAX_NOPROGRESS ]]; then
            log "ABORT: $noprog consecutive no-progress attempts. Config may be stuck/divergent — stopping."
            exit 1
        fi
    fi

    sleep "$BACKOFF"
done
