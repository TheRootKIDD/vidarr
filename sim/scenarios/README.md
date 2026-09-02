# sim/scenarios/

One YAML per hardware scenario. **Every value carries a source line.** For
`local_3060.yaml` the source is a measured result id under `experiments/`; for
`note64` and `gpu_today` it is a vendor document or a published figure, or the
value is explicitly labelled a placeholder (open question I4).

Hardware numbers are scenario inputs, never constants in code (CLAUDE.md).
