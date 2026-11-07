# 048-l8b-pv-sizing — L8b-pv sizing and equivalence (ADR-060 §5)

**Bears on:** no hypothesis (rig series). It checks ADR-060's premise and sizes `178`/`179`.

## Equivalence on real shapes (`equivalence.json`, `pv_equiv.py`)

`176`'s trained L8b weights were loaded into both code paths at its width, 704. The batch was 16 held-out FineWeb-Edu windows of 2,048 tokens with their true neighbours from the pipeline.

| precision | max \|loss(L8b-pv) − loss(L8b)\| over 4 batches |
|---|---|
| fp32 | **2.4 × 10⁻⁷** nats |
| bf16 autocast (training precision) | 4.0 × 10⁻⁴ nats |

In fp32 the two paths are the same function, up to float rounding. In bf16 the reordering (sum the projected vectors, against project the sum) moves the per-batch loss by ≤ 4 × 10⁻⁴. That is rounding noise, a seventh of σ and below anything the pair rule reads. The unit test (`test_l8b_pv_equals_l8b_forward_and_backward`) covers gradients at toy size.

## Sizing (`sizing/`, 6 DDP steps at 2.5 B-token schedule)

| | L8b-pv (`048`) | L8b (`045` / `176`) |
|---|---|---|
| expert width | **832** | 704 |
| peak memory per GPU | 5.78 GiB | 5.72 GiB |
| throughput, steps 2–6 | 33.6–34.4 k tok/s | 33.8 k (`045`), 34.1 k (`176`, median) |
| step | 15.2–15.6 s | 15.4 s |
| retrieval per micro-batch | 0.31–0.34 s | 0.31–0.32 s |
| full held-out eval | 158 s | 158 s |

The wider experts cost no throughput, because the hoisted maps save at least what they add. Retrieval (encoder and search) is unchanged and still dominates the step's overhead.

**Estimate:** ≈ 20.3 h of training plus ≈ 0.4 h of in-training evals per run, so ≈ 20.8 h each. The pair takes ≈ 42 h. `queue_small_12` started `178` at 15:34 on `3a4e219`.

**Recorded:** the random-rows probe now draws its rows in `prepare`. Under L8b-pv one draw serves every iteration of a segment, whereas L8b re-draws at each iteration. The probe means the same thing (true neighbours against random rows), but its numbers are not draw-for-draw comparable with `176`/`177`'s.
