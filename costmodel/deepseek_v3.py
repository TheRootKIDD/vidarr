"""DeepSeek-V3's shape for calibration point (b) (ADR-052 §3): parameters, decode FLOPs and bytes.

A layered MoE, not the recurrent design: it exists so that `sim/` can be scored against a
published deployment (SGLang's EP72 decode, I1). The shape is from the V3 technical report
(arXiv:2412.19437 §2, §4.2) and the released `config.json` of `deepseek-ai/DeepSeek-V3`:
61 layers (the first 3 dense), d = 7168, 256 routed experts of width 2048 plus 1 shared,
top-8, MLA with q-LoRA rank 1536, KV-LoRA rank 512, 64 rope dimensions, 128 heads of
128 (nope) / 128 (value), vocabulary 129,280, untied embedding and head. The MTP module is
not part of the decode path (the scored run used none).

Decode attention is in MLA's absorbed form: the cache holds the 512-wide latent plus the
64-wide rope key per token per layer, W_UK is folded into the query and W_UV into the output,
so each head scores against the 576-wide entry and reads 512-wide values.

Pure functions; numerics (bytes per element) are arguments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV3:
    d: int = 7168
    layers: int = 61
    dense_layers: int = 3  # first_k_dense_replace
    d_ff_dense: int = 18432  # intermediate_size
    n_routed: int = 256
    n_shared: int = 1
    d_ff_expert: int = 2048  # moe_intermediate_size
    k: int = 8  # num_experts_per_tok
    heads: int = 128
    q_lora: int = 1536
    kv_lora: int = 512
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    vocab: int = 129_280

    @property
    def moe_layers(self) -> int:
        return self.layers - self.dense_layers

    @property
    def kv_entry(self) -> int:
        """Cached values per token per layer: latent plus rope key (576)."""
        return self.kv_lora + self.qk_rope


def swiglu(d: int, d_ff: int) -> int:
    return 3 * d * d_ff


def attention_gemm_params(m: DeepSeekV3) -> int:
    """The token-wise projections of one MLA layer: q_a, q_b, kv_a (latent + rope key), o."""
    q_a = m.d * m.q_lora
    q_b = m.q_lora * m.heads * (m.qk_nope + m.qk_rope)
    kv_a = m.d * m.kv_entry
    o = m.heads * m.v_head * m.d
    return q_a + q_b + kv_a + o


def attention_absorbed_params(m: DeepSeekV3) -> int:
    """kv_b (W_UK and W_UV), applied per head in the absorbed form: 512 x (128 + 128) x heads."""
    return m.kv_lora * m.heads * (m.qk_nope + m.v_head)


def attention_params(m: DeepSeekV3) -> int:
    return attention_gemm_params(m) + attention_absorbed_params(m)


def router_params(m: DeepSeekV3) -> int:
    return m.n_routed * m.d


def total_params(m: DeepSeekV3) -> int:
    """Every weight (norms omitted, < 0.01 %): ≈ 671 B, the report's figure without MTP."""
    attn = m.layers * attention_params(m)
    dense = m.dense_layers * swiglu(m.d, m.d_ff_dense)
    moe = m.moe_layers * ((m.n_routed + m.n_shared) * swiglu(m.d, m.d_ff_expert) + router_params(m))
    return attn + dense + moe + 2 * m.vocab * m.d


def active_params(m: DeepSeekV3) -> int:
    """Weights one token touches: ≈ 37 B (the report's "activated"), embedding row excluded."""
    attn = m.layers * attention_params(m)
    dense = m.dense_layers * swiglu(m.d, m.d_ff_dense)
    moe = m.moe_layers * ((m.k + m.n_shared) * swiglu(m.d, m.d_ff_expert) + router_params(m))
    return attn + dense + moe + m.vocab * m.d


def attention_proj_flops(m: DeepSeekV3) -> float:
    """Per decode token per layer: the projections plus the absorbed W_UK / W_UV products."""
    return 2.0 * attention_params(m)


def attention_score_flops_per_key(m: DeepSeekV3) -> float:
    """Per decode token per layer per cached key: q·k over 576 and p·v over 512, every head."""
    return 2.0 * m.heads * (m.kv_entry + m.kv_lora)


def kv_bytes_per_key(m: DeepSeekV3, b_kv: float) -> float:
    return m.kv_entry * b_kv


def attention_weight_bytes(m: DeepSeekV3, b_gemm: float, b_absorbed: float) -> float:
    """One layer's attention weights read per batched step (GEMMs in `b_gemm`, kv_b in
    `b_absorbed`)."""
    return attention_gemm_params(m) * b_gemm + attention_absorbed_params(m) * b_absorbed


def expert_flops(m: DeepSeekV3) -> float:
    """One token through one expert (routed or shared): 2 · 3 d d_ff."""
    return 2.0 * swiglu(m.d, m.d_ff_expert)


def dense_ff_flops(m: DeepSeekV3) -> float:
    return 2.0 * swiglu(m.d, m.d_ff_dense)


def head_flops(m: DeepSeekV3) -> float:
    return 2.0 * m.vocab * m.d
