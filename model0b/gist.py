"""AHN-style gist: a gated delta-rule state per head, written with tokens that leave the window.

Recurrence per token (S is d_v x d_k, k L2-normalized):
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
Reads in chunk j use the state as of the start of chunk j (chunk-level causality), so only the
state at block boundaries is needed. `block_update` computes it for a whole block at once with the
WY / UT form (see block_terms): with gamma_t = prod_{i<=t} alpha_i and u_t = beta_t (v_t - alpha_t S_{t-1} k_t),
    (I + L) U = diag(beta) V - diag(beta gamma) K S_0^T,  L_ti = beta_t (gamma_t / gamma_i) k_t.k_i  (i < t)
    S_c = gamma_c S_0 + U^T diag(gamma_c / gamma) K.
"""
import torch


def naive_update(S, K, V, log_alpha, beta):
    """Reference: per-token loop. S (..., dv, dk), K (..., c, dk), V (..., c, dv), log_alpha/beta (..., c)."""
    eye = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    for t in range(K.shape[-2]):
        k, v = K[..., t, :], V[..., t, :]
        a, b = log_alpha[..., t].exp()[..., None, None], beta[..., t][..., None, None]
        S = a * S @ (eye - b * k[..., :, None] * k[..., None, :]) + b * v[..., :, None] * k[..., None, :]
    return S


def block_terms(K, V, log_alpha, beta):
    """For each block (leading dims arbitrary, block of c tokens): M (dk x dk) and P (dv x dk) with
    S_end = S_0 M + P. The triangular solve does not depend on S_0, so all blocks are solved at once:
    with T = (I + L)^{-1}, X = T diag(beta) V and Y = T diag(beta gamma) K,
    U = X - Y S_0^T, hence M = gamma_c I - Y^T diag(w) K and P = X^T diag(w) K, w = gamma_c / gamma."""
    c, dk = K.shape[-2], K.shape[-1]
    lg = torch.cumsum(log_alpha, dim=-1)                                   # log gamma_t
    diff = lg[..., :, None] - lg[..., None, :]                             # log(gamma_t / gamma_i)
    strict = torch.ones(c, c, dtype=torch.bool, device=K.device).tril(-1)
    decay = torch.where(strict, diff, torch.full_like(diff, -float("inf"))).exp()
    A = torch.eye(c, dtype=K.dtype, device=K.device) + beta[..., :, None] * decay * (K @ K.transpose(-1, -2))
    rhs = torch.cat([beta[..., None] * V, (beta * lg.exp())[..., None] * K], dim=-1)
    XY = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
    X, Y = XY[..., :V.shape[-1]], XY[..., V.shape[-1]:]
    wK = (lg[..., -1:] - lg).exp()[..., None] * K
    eye = torch.eye(dk, dtype=K.dtype, device=K.device)
    M = lg[..., -1].exp()[..., None, None] * eye - Y.transpose(-1, -2) @ wK
    return M, X.transpose(-1, -2) @ wK


def block_update(S, K, V, log_alpha, beta):
    """Same result as naive_update, computed for the whole block at once."""
    M, P = block_terms(K, V, log_alpha, beta)
    return S @ M + P


def read_states(K, V, log_alpha, beta, chunk, window, update=block_update):
    """Gist state readable by each chunk.

    K, V: (B, H, T, d) with T a multiple of `chunk`, and `window` a multiple of `chunk`.
    Block b (tokens [b c, (b+1) c)) leaves the window after chunk b + window/c, so chunk j reads the
    state after blocks 0 .. j - window/c - 1. Returns (B, H, n + 1, dv, dk): entry j for chunk j,
    entry n for a segment that follows the sequence (evaluation probes).
    """
    B, H, T, dk = K.shape
    n, lag = T // chunk, window // chunk
    m = max(0, n - lag)                                                    # blocks that leave the window
    S = K.new_zeros(B, H, V.shape[-1], dk)
    states = [S]
    if update is block_update and m:
        blk = lambda x: x[:, :, :m * chunk].reshape(B, H, m, chunk, *x.shape[3:])
        M, P = block_terms(blk(K), blk(V), blk(log_alpha), blk(beta))       # all blocks in parallel
        for b in range(m):
            S = S @ M[:, :, b] + P[:, :, b]
            states.append(S)
    else:
        for b in range(m):
            sl = slice(b * chunk, (b + 1) * chunk)
            S = update(S, K[:, :, sl], V[:, :, sl], log_alpha[:, :, sl], beta[:, :, sl])
            states.append(S)
    idx = [max(0, j - lag) for j in range(n + 1)]
    return torch.stack([states[i] for i in idx], dim=2)
