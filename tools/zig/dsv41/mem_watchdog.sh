#!/usr/bin/env bash
# Host memory watchdog for a DGX Spark (unified memory shared by CPU and GPU).
# Every second it reads MemAvailable; when it falls below $MIN_AVAIL_GIB (default 3: the DSV4.1 goal configuration runs at 5.0 GiB free on rank 0 and dips to 7.5 while loading) it kills every
# docker container except those matching $KEEP (default sparkDash) and logs what it did. A node whose
# memory is exhausted stops answering ssh and needs a power cycle, so the watchdog errs on the side
# of killing work. Run one per node in the background:
#   nohup bash mem_watchdog.sh > ~/mem_watchdog.log 2>&1 &
# `bash mem_watchdog.sh status` prints whether one is running.
set -u
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-3}
KEEP=${KEEP:-sparkDash}
PIDFILE=${PIDFILE:-$HOME/.mem_watchdog.pid}
if [ "${1:-}" = status ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "running pid $(cat "$PIDFILE")"; exit 0; fi
  echo "not running"; exit 1
fi
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "already running pid $(cat "$PIDFILE")"; exit 0; fi
echo $$ > "$PIDFILE"
min_kib=$((MIN_AVAIL_GIB * 1024 * 1024))
echo "$(date +%FT%T) watchdog start: kill containers when MemAvailable < ${MIN_AVAIL_GIB} GiB (keep: $KEEP)"
while true; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$min_kib" ]; then
    names=$(docker ps --format '{{.Names}}' | grep -v -E "^($KEEP)$" || true)
    echo "$(date +%FT%T) MemAvailable ${avail} KiB < ${min_kib} KiB: killing: $(echo $names | tr '\n' ' ')"
    for n in $names; do docker kill "$n" >/dev/null 2>&1 || true; done
    for p in $(pgrep -f 'ncu|compute-sanitizer' || true); do kill -9 "$p" 2>/dev/null || true; done
    sleep 5
  fi
  sleep 1
done
