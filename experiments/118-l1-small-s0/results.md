# 118-l1-small-s0 — killed at step 390 for a planned power outage; partial record only

`python -m scripts.train.train --rung L1 --id 118-l1-small-s0 --tokens 2.5e+09 --micro-batch 2 --seed 0` ·
started 2026-09-16 06:53 by `queue_small_1.sh`, **stopped by hand at ≈ 08:05** (1.2 h in) because the
user announced a mains power outage 20 minutes ahead. The queue script was stopped first so that it
would not start `119`; the trainer's four DDP workers had to be killed by PID after the parent
(`pkill` of the parent leaves the spawned workers training at full power, as `017`/`019` saw).

**Bears on:** nothing. Ids are append-only and checkpoints are not resumed (`docs/04` 2026-09-14),
so this id is burnt; **L1 seed 0 at `small` reruns as `130-l1-small-s0`** in `queue_small_2.sh`.

Last logged step: `step   400/4768 tok  0.210B loss 4.1159 main 4.1033 lr 5.94e-04 gn 0.32   50,209 tok/s 8.72 GiB`

50.2 k tok/s at 8.72 GiB per GPU (micro-batch 2), zero `n_thermal` rows over 1.2 h. The 2 M-token
subset eval at step 300 was 4.5053 nats. No number here is used anywhere.
