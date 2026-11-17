"""Gated delta rule with per-token outputs, pure PyTorch (the fla kernel AHN uses needs a forked package).

Per head, with S (dv x dk), unit-norm k and q:
    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   o_t = scale * S_t q_t
(fla's chunk_gated_delta_rule with use_qk_l2norm_in_kernel and its default scale dk^-1/2, as AHN calls it).
WY form within a block of c tokens (gamma_t = prod_{i<=t} alpha_i, u_t = beta_t (v_t - alpha_t S_{t-1} k_t)):
    (I + L) U = diag(beta) V - diag(beta gamma) K S_0^T,  L_ti = beta_t (gamma_t / gamma_i) k_t.k_i  (i < t)
    S_t = gamma_t S_0 + sum_{i<=t} (gamma_t / gamma_i) u_i k_i^T, so
    O = gamma * (Q S_0^T) + (D * Q K^T) U with D_ti = gamma_t / gamma_i for i <= t.
"""
import torch


def naive_outputs(S, K, V, Q, log_alpha, beta, scale):
    eye = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    out = []
    for t in range(K.shape[-2]):
        k, v = K[..., t, :], V[..., t, :]
        a, b = log_alpha[..., t].exp()[..., None, None], beta[..., t][..., None, None]
        S = a * S @ (eye - b * k[..., :, None] * k[..., None, :]) + b * v[..., :, None] * k[..., None, :]
        out.append(scale * (S @ Q[..., t, :, None])[..., 0])
    return torch.stack(out, -2), S


def block_outputs(S0, K, V, Q, log_alpha, beta, scale):
    """S0 (..., dv, dk); K, Q (..., c, dk) unit norm; V (..., c, dv); log_alpha, beta (..., c).
    Returns (O (..., c, dv), S_end (..., dv, dk)); equal to naive_outputs."""
    c = K.shape[-2]
    lg = torch.cumsum(log_alpha, dim=-1)
    diff = lg[..., :, None] - lg[..., None, :]
    tri = torch.ones(c, c, dtype=torch.bool, device=K.device)
    strict = tri.tril(-1)
    decay_s = torch.where(strict, diff, torch.full_like(diff, -float("inf"))).exp()
    decay_i = torch.where(tri.tril(), diff, torch.full_like(diff, -float("inf"))).exp()
    A = torch.eye(c, dtype=K.dtype, device=K.device) + beta[..., :, None] * decay_s * (K @ K.transpose(-1, -2))
    rhs = beta[..., None] * V - (beta * lg.exp())[..., None] * (K @ S0.transpose(-1, -2))
    U = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)          # (..., c, dv)
    O = lg.exp()[..., None] * (Q @ S0.transpose(-1, -2)) + (decay_i * (Q @ K.transpose(-1, -2))) @ U
    w = (lg[..., -1:] - lg).exp()[..., None]
    S_end = lg[..., -1].exp()[..., None, None] * S0 + U.transpose(-1, -2) @ (w * K)
    return scale * O, S_end
