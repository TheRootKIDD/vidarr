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
# Killing the launcher alone leaves the torchrun children and the multiprocessing
# workers training at full power on every card (seen in 017 and 019), so the trap
# takes the whole run down by id, not just the sampler.
cleanup() { kill $SAMPLER 2>/dev/null; pkill -f "bench_train_step --id $ID" 2>/dev/null; }
trap cleanup EXIT INT TERM
.venv/bin/python -m scripts.bench.bench_train_step --id "$ID" --ddp --steps "$STEPS" --warmup 10 --rung L5:4 \
  2>&1 | grep -v -i warning | tee "$OUT/soak.log"
kill $SAMPLER 2>/dev/null; trap - EXIT INT TERM
echo "--- summary (last 5 min per GPU: max temp, min clock, thermal vs power-cap samples)"
python3 - "$OUT/thermals.csv" <<'PY'
import csv, sys, collections
# NVML packs every throttle reason into one bitmask and they are not equivalent.
# I23 is about *thermal* slowdown. Hitting the 170 W software power cap is the card
# working as configured (015: these cards are thermally, not power, limited), and a
# well-cooled card reaches it more often, not less -- counting the two together makes
# better cooling read as more throttling, which is how 024 first reported 3/30 on a
# card with zero thermal events. The acceptance criterion reads `thermal`.
SW_POWER_CAP = 0x4
THERMAL = 0x20 | 0x40  # SwThermalSlowdown | HwThermalSlowdown
rows = list(csv.DictReader(open(sys.argv[1])))
tail = rows[-4 * 30:]  # ~5 min at 10 s per sample, 4 GPUs
by = collections.defaultdict(list)
for r in tail: by[r["idx"]].append(r)
for i, rs in sorted(by.items()):
    temps = [float(r["temp_c"]) for r in rs]; clk = [float(r["sm_mhz"]) for r in rs]
    reasons = [int(r["throttle_hex"], 16) for r in rs]
    thermal = sum(bool(v & THERMAL) for v in reasons)
    powercap = sum(bool(v & SW_POWER_CAP) for v in reasons)
    verdict = "PASS" if thermal == 0 and max(temps) < 85 else "FAIL"
    print(f"  GPU{i}: max {max(temps):.0f} C, clock min {min(clk):.0f} / mean {sum(clk)/len(clk):.0f} MHz, "
          f"thermal {thermal}/{len(rs)}, power-cap {powercap}/{len(rs)}  [{verdict}]")
PY
