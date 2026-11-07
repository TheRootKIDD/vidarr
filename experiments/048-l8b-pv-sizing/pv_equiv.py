"""Rig 048 (ADR-060 §5): L8b-pv = L8b on real shapes, with 176's trained weights."""
import dataclasses as dc, json, sys, time
from pathlib import Path
import torch
from model.model import BigMoE
from scripts.analysis.probe_routing import RESULTS_DIR, load_run
from scripts.analysis.eval_wiki_lead import CORPUS
from scripts.train.data import MemmapCorpus
dev = torch.device("cuda:0")
t0 = time.time()
base, cfg, step = load_run(RESULTS_DIR / "176-l8b-small-s0", dev)
# width pinned at 176s 704, so only the compute order differs
pv = BigMoE(dc.replace(cfg, memory=dc.replace(cfg.memory, hoist=True), experts=dc.replace(cfg.experts, d_ff=base.cfg.expert_width())))
pv.load_state_dict(base.state_dict())
pv.middle.mem.encoder = base.middle.mem.encoder
pv = pv.to(dev).eval()
wins = MemmapCorpus(Path(CORPUS)).val_windows(cfg.context, max_tokens=16 * (cfg.context + 1))
toks = torch.cat([w.reshape(-1, w.shape[-1]) for w in wins])[:16].to(dev)
res = {"run": "176-l8b-small-s0", "step": step, "windows": len(toks), "per_batch": []}
for prec in ("bf16", "fp32"):
  base.float() if prec == "fp32" else None
  pv.float() if prec == "fp32" else None
  with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
    for i in range(0, len(toks), 4):
        x = toks[i : i + 4]
        nbr = base.middle.mem.neighbours(x[:, :-1])
        la = base.loss(x, neighbours=nbr).loss.float().item()
        lb = pv.loss(x, neighbours=nbr).loss.float().item()
        res["per_batch"].append({"prec": prec, "l8b": la, "l8b_pv": lb, "diff": lb - la})
for prec in ("bf16", "fp32"):
    res["max_abs_diff_" + prec] = max(abs(r["diff"]) for r in res["per_batch"] if r["prec"] == prec)
res["wall_s"] = time.time() - t0
out = Path(sys.argv[1]); out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
