"""Pilot slot cortex: consolidation, the interference gate, and anchors (CPU, no model needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pilot.schema_memory import SchemaMemory  # noqa: E402


def bare(G=1, C=8, A=4, d=16, theta=0.9, rho=4.0, eta=1.0):
    m = object.__new__(SchemaMemory)
    z = lambda *s: torch.zeros(*s)
    m.theta, m.rho, m.eta, m.G, m.A_cap = theta, rho, eta, G, A
    m.Ck, m.Cv, m.Cuse = z(G, C, d), z(G, C, d), z(G, C)
    m.Cval = torch.zeros(G, C, dtype=torch.bool)
    m.Ak, m.Av, m.Aval, m.A_seen = z(G, A, d), z(G, A, d), np.zeros(A, bool), 0
    m.rng = np.random.default_rng(0)
    m.st = dict(steps_shrunk=0, steps_rejected=0)
    return m


def unit(d, i):
    x = torch.zeros(1, 1, d); x[0, 0, i] = 1.0
    return x


def test_consolidate_then_recall_exactly():
    m = bare()
    k, v = unit(16, 0), unit(16, 5) * 3
    m.consolidate(k, v, [0], eta=1.0, gate=False)
    assert m._err(k, v).item() < 1e-6                                     # C(k) = v after one full step
    assert m._err(unit(16, 1), v).item() == pytest.approx(1.0)              # unrelated key: C(k) = 0


def test_gate_rejects_correlated_conflict_accepts_unrelated():
    m = bare(theta=0.9, rho=4.0)
    k_rule, v_rule = unit(16, 0), unit(16, 5)
    m.consolidate(k_rule, v_rule, [0], eta=1.0, gate=False)
    m._add_anchors(k_rule, torch.ones(1, dtype=torch.bool), [0])            # anchor: v_a = C(k_rule) = v_rule
    k_exc = k_rule + 0.1 * unit(16, 1)                                     # cosine 0.995 with the rule key
    acc = m.consolidate(k_exc, unit(16, 6), [0], eta=1.0, gate=True)        # conflicting value: would overwrite rule
    # eta 1 and 1/2 fail G - 4 D > 0; eta 1/4 passes (gain 0.875 > 4 * 0.125), so the rule is only slightly damaged
    assert acc.all() and m.st["steps_shrunk"] == 1
    assert m._err(k_rule, v_rule).item() == pytest.approx(0.125, abs=1e-5)
    m.rho = 100.0
    acc = m.consolidate(k_exc, unit(16, 6), [0], eta=1.0, gate=True)        # strict gate: rejected outright
    assert not acc.any() and m._err(k_rule, v_rule).item() == pytest.approx(0.125, abs=1e-5)
    m.rho = 4.0
    acc = m.consolidate(unit(16, 2), unit(16, 7), [0], eta=1.0, gate=True)  # unrelated key: free slot, no damage
    assert acc.all() and m._err(unit(16, 2), unit(16, 7)).item() < 1e-6
    acc = m.consolidate(k_exc, unit(16, 6), [0], eta=1.0, gate=False)       # without the gate the rule is damaged
    assert m._err(k_rule, v_rule).item() > 0.5


def test_full_cortex_reuses_least_used_slot():
    m = bare(C=2)
    m.consolidate(torch.cat([unit(16, 0), unit(16, 1)], 1), torch.cat([unit(16, 4), unit(16, 5)], 1), [0], 1.0, False)
    m.Cuse[0] = torch.tensor([5.0, 0.0])
    m.consolidate(unit(16, 2), unit(16, 6), [0], 1.0, False)
    assert m._err(unit(16, 0), unit(16, 4)).item() < 1e-6                 # well-used slot kept
    assert m._err(unit(16, 2), unit(16, 6)).item() < 1e-6                 # least-used slot replaced


def test_linear_cortex_gate_matches_proposition_1():
    """Linear cortex replay step: the computed gain and damage equal Proposition 1's closed form."""
    from pilot.linear_cortex import LinearSchemaMemory
    m = object.__new__(LinearSchemaMemory)
    g = torch.Generator().manual_seed(0)
    d, nB, nA = 8, 5, 6
    W = torch.randn(1, d, d, generator=g) * 0.3
    KB = torch.nn.functional.normalize(torch.randn(1, nB, d, generator=g), dim=-1)
    VB = torch.nn.functional.normalize(torch.randn(1, nB, d, generator=g), dim=-1)
    KA = torch.nn.functional.normalize(torch.randn(1, nA, d, generator=g), dim=-1)
    VA = torch.einsum("gde,gne->gnd", W, KA)                         # anchors fitted: R_A = 0
    grad = m._grad(W, KB, VB)
    eta = 0.7
    W2 = W - eta * grad
    gain = (m._loss(W, KB, VB) - m._loss(W2, KB, VB)) * nB
    dmg = (m._loss(W2, KA, VA) - m._loss(W, KA, VA)) * nA
    # Prop. 1 with sum losses (0.5 * sum of squares) and step eta' = eta / nB on R_B K_B^T
    R = (torch.einsum("gde,gne->gnd", W, KB) - VB)[0].T             # (d, nB)
    K, Ka = KB[0].T, KA[0].T
    e = eta / nB
    Mbb, Mba = K.T @ K, K.T @ Ka
    G1 = e * (R * (R @ Mbb)).sum() - e ** 2 / 2 * ((R @ Mbb) ** 2).sum()
    D1 = e ** 2 / 2 * ((R @ Mba) ** 2).sum()
    assert torch.allclose(gain[0], G1, atol=1e-5) and torch.allclose(dmg[0], D1, atol=1e-5)


def test_lora_linear_identity_until_trained():
    from pilot.lora_linear import LoRALinear
    base = torch.nn.Linear(6, 4)
    lo = LoRALinear(base, r=2)
    x = torch.randn(3, 6)
    assert torch.allclose(lo(x), base(x))                       # B = 0 at init: no change to the model
    lo.B.data.normal_()
    assert not torch.allclose(lo(x), base(x))
    lo.enabled = False
    assert torch.allclose(lo(x), base(x))                       # disabled: exactly the frozen layer


def test_lora_memory_episode_ids_resolve_to_items():
    """Regression: queued arrival/surprise episodes had tuple ids, which _lm_nll took for (tokens, offsets) items."""
    from pilot.lora_cortex import LoRAMemory
    m = object.__new__(LoRAMemory)
    m.ep_tok, m.ep_lm = {3: [5, 6, 7], -11: [8, 9]}, {3: [1, 2], -11: [1]}
    assert m._items([3, -11]) == [([5, 6, 7], [1, 2]), ([8, 9], [1])]
    assert m._items([([1, 2], [1])]) == [([1, 2], [1])]


def test_hrd_targets_keep_confirmed_recalls_only():
    """Regression (4 Nov 2026): targets were ranked by KL(teacher || base) alone, so generic context shifts and recalls
    the stream contradicted became targets. Now a target must raise the log-prob of the token that actually came next."""
    from pilot.lora_cortex import hrd_targets
    L, V = 14, 10
    b = torch.log_softmax(torch.zeros(L, V), -1)
    t = b.clone()
    nxt = torch.full((L - 1,), 5)
    peak = lambda tok: torch.log_softmax(torch.nn.functional.one_hot(torch.tensor(tok), V).float() * 6.0, -1)
    t[3] = peak(5)                       # confirmed but too little query context (< min_ctx)
    t[9], t[11] = peak(5), peak(5)       # confirmed recalls: the episode predicted what came next
    t[10] = peak(2)                      # big KL shift, but the stream said 5: contradicted, not a target
    t[13] = peak(5)                      # last position: no next token to confirm
    out = hrd_targets(t, b, nxt, n_pos=4, min_ctx=8, topk=4, g_min=1.0)
    assert sorted(out["pos"].tolist()) == [9, 11]
    assert out["t_ids"].shape == (2, 4) and torch.allclose(out["t_logp"].exp().sum(-1), torch.ones(2))
    assert hrd_targets(t, b, nxt, min_ctx=8, g_min=100.0)["pos"] is None   # the episode never helped enough
    assert hrd_targets(t[:9], b[:9], nxt[:8], min_ctx=8)["pos"] is None    # query too short


def test_kl_topk_zero_iff_student_matches_teacher():
    from pilot.lora_cortex import kl_topk
    V = 10
    p = torch.log_softmax(torch.randn(3, V), -1)
    ti = torch.arange(V).expand(3, V)
    assert torch.allclose(kl_topk(ti, p, p), torch.zeros(3), atol=1e-6)
    assert (kl_topk(ti, p, torch.log_softmax(torch.randn(3, V), -1)) > 0).all()


def test_hrd_on_cue_records_one_query_per_episode_per_chunk(monkeypatch):
    import sys
    import types
    from pilot.lora_cortex import LoRAMemory
    monkeypatch.setitem(sys.modules, "pilot.engine", types.SimpleNamespace(_CTX={"st": {"tokens": np.arange(100)}}))
    m = object.__new__(LoRAMemory)
    m.target, m.n_q, m.q_pre, m.q_post = "hrd", 2, 3, 1
    m.ep_tok, m.ep_q, m.st = {0: [1], 1: [2]}, {}, dict(hrd_queries=0)
    m.Hep = np.array([0, 0, 1, -1])
    hit = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)  # landmark 20 re-cues ep 0; 22 re-cues 0, 1
    m._on_cue(np.array([20, 22]), hit)
    assert [q["ids"] for q in m.ep_q[0]] == [list(range(17, 22))]       # once per chunk, from the first landmark
    assert [q["ids"] for q in m.ep_q[1]] == [list(range(19, 24))]
    m._on_cue(np.array([40]), hit[:1]); m._on_cue(np.array([60]), hit[:1])
    assert [q["ids"][0] for q in m.ep_q[0]] == [37, 57]                  # last n_q kept
    assert m.st["hrd_queries"] == 4
    m.target = "lm"
    m._on_cue(np.array([80]), hit[:1])
    assert m.st["hrd_queries"] == 4                                      # the LM-target arms record nothing


def test_hrd_recall_hook_counts_confirmed_copies_only(monkeypatch):
    import sys
    import types
    from pilot.lora_cortex import LoRAMemory
    toks = np.arange(100) % 50
    toks[5], toks[51] = 77, 77                    # landmark at stream position 5 holds token 77; stream says 77 at 51
    monkeypatch.setitem(sys.modules, "pilot.engine", types.SimpleNamespace(_CTX={"st": {"tokens": toks}}))
    m = object.__new__(LoRAMemory)
    m.reuse, m.recall_from, m.w_min, m.q_pre, m.q_post, m.n_q = "recall", 0, 0.3, 4, 16, 4
    m.last, m.last_top = {0: ["H"]}, {0: [torch.tensor([[[2, 0], [0, 2]]])]}   # (B, Hkv, M): KV head 0 reads 2, 0
    m.Hval, m.Hpos, m.Hep = np.array([True, False, True]), np.array([9, -1, 5]), np.array([3, -1, 7])
    m.Hcue, m.ep_tok, m.ep_q, m.st = torch.zeros(3), {7: [1], 3: [1]}, {}, dict(hrd_queries=0)
    m._rec_pos, m._rec_ep = set(), set()
    pos = np.arange(48, 56)                                     # chunk positions 48..55, T = 8
    wm = torch.zeros(1, 2, 8, 2)                                # (B, Hq, T, M), one query head per KV head
    wm[0, 0, 2, 0] = 0.9                                        # p = 50 reads slot 2 (token 77); next token 77: recall
    wm[0, 0, 4, 0] = 0.9                                        # p = 52 reads slot 2; next token is not 77: no
    wm[0, 1, 5, 1] = 0.2                                        # p = 53 reads slot 2 weakly: below w_min
    wm[0, 0, 7, 0] = 0.9                                        # last position of the chunk: next token unseen
    m.recall_hook(0, wm, pos)
    m.recall_hook(0, wm, pos)                                   # a second layer seeing the same recall counts once
    assert m.Hcue.tolist() == [0.0, 0.0, 1.0] and m.st["recalls"] == 1
    assert list(m.ep_q) == [7] and m.st["hrd_queries"] == 1
    q = m.ep_q[7][0]
    assert q["start"] == 46 and q["ids"] == toks[46:56].tolist()  # clipped at the chunk end (55), not p + q_post
