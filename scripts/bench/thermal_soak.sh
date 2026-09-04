#!/usr/bin/env bash
# Thermal soak (I23): run the L5 DDP training step for ~15 min at the current
# power limit while sampling every GPU every 10 s. Writes
# experiments/<id>/{thermals.csv,soak.log}; the caller sets the power limit
# beforehand (root): sudo nvidia-smi -pl <watts>.
#   scripts/bench/thermal_soak.sh 017-thermal-soak-<config> [steps]
# The run is step-bound: 1100 steps is ~15 min at the un-throttled L5 step of
# ~0.8 s (015 and 016 used 350, which is 5-9 min depending on throttling).
set -euo pipefail
cd "$(dirname "$0")/../.."
ID=$1; STEPS=${2:-1100}
OUT=experiments/$ID; mkdir -p "$OUT"
[ -e "$OUT/thermals.csv" ] && { echo "refusing to overwrite $OUT (append-only)"; exit 1; }
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader > "$OUT/power_limit.txt"
echo "time,idx,temp_c,sm_mhz,power_w,fan_pct,throttle_hex" > "$OUT/thermals.csv"
( while true; do
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,fan.speed,clocks_throttle_reasons.active \
      --format=csv,noheader,nounits | sed "s/^/$(date +%T),/" | tr -d ' ' >> "$OUT/thermals.csv"
    sleep 10
  done ) & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
.venv/bin/python -m scripts.bench.bench_train_step --id "$ID" --ddp --steps "$STEPS" --warmup 10 --rung L5:4 \
  2>&1 | grep -v -i warning | tee "$OUT/soak.log"
kill $SAMPLER 2>/dev/null; trap - EXIT
echo "--- summary (last 5 min per GPU: max temp, min clock, throttled samples)"
python3 - "$OUT/thermals.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
tail = rows[-4 * 30:]  # ~5 min at 10 s per sample, 4 GPUs
by = collections.defaultdict(list)
for r in tail: by[r["idx"]].append(r)
for i, rs in sorted(by.items()):
    temps = [float(r["temp_c"]) for r in rs]; clk = [float(r["sm_mhz"]) for r in rs]
    thr = sum(int(r["throttle_hex"], 16) not in (0, 1) for r in rs)
    print(f"  GPU{i}: max {max(temps):.0f} C, clock min {min(clk):.0f} / mean {sum(clk)/len(clk):.0f} MHz, throttled {thr}/{len(rs)} samples")
PY
