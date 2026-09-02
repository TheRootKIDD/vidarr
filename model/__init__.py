"""Reference implementation of `docs/01` (Phase 1). Readability over speed."""

from model.config import ModelCfg, ladder
from model.model import BigMoE, LossOut

__all__ = ["BigMoE", "LossOut", "ModelCfg", "ladder"]
