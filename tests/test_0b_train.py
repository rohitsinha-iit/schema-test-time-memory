"""Stage 0b training utilities."""
import pytest

pytest.importorskip("torch")

from train0b import lr_at  # noqa: E402


def test_constant_schedule_warms_up_then_stays():
    lrs = [lr_at(s, 1000, 1e-3, 100, "constant") for s in range(1000)]
    assert lrs[0] == pytest.approx(1e-5) and lrs[99] == pytest.approx(1e-3)
    assert all(x == 1e-3 for x in lrs[100:])
    cos = [lr_at(s, 1000, 1e-3, 100) for s in range(1000)]
    assert cos[100] == pytest.approx(1e-3) and cos[-1] < 1e-7          # default keeps the cosine decay


def test_reference_gpt_shapes_and_param_groups():
    import torch
    from model0b.reference import GPT
    m = GPT(vocab=50, d_model=32, n_layers=2, n_heads=4, max_len=64)
    assert m(torch.randint(0, 50, (2, 64))).shape == (2, 64, 50)
    emb = {id(x.weight) for x in m.modules() if isinstance(x, torch.nn.Embedding)}
    assert len(emb) == 2                                               # token and learned position tables


def test_shortcut_hits_on_handmade_sequence():
    import numpy as np
    from eval0b import shortcut_hits
    from sim0b.stream import BOS, K0, QRY, SEP, V0
    a, b, c = K0 + 1, K0 + 2, K0 + 3
    x, y = V0 + 5, V0 + 6
    # facts a->x, b->y, a->x, c->y; probe "QRY b ?" (answer y) must not count as a fact
    t = np.array([BOS, a, x, SEP, b, y, SEP, a, x, SEP, QRY, b, y, c, y, SEP, QRY, a, x])
    rec, frq = shortcut_hits(t, [12, 14, 18])
    # pos 12 (y): recent fact value x, most frequent x -> both wrong
    # pos 14 (y, fact c): recent fact x (probe y not counted), freq x -> wrong; pos 18 (x): recent y, freq tie x/y -> y
    assert rec.tolist() == [False, False, False] and frq.tolist() == [False, False, False]
    rec, frq = shortcut_hits(t, [8])                                   # second a->x: recent y, freq tie x/y -> y
    assert rec.tolist() == [False] and frq.tolist() == [False]
    rec, frq = shortcut_hits(t, [5])                                   # b->y after one fact x: both predict x
    assert rec.tolist() == [False] and frq.tolist() == [False]
    t2 = np.array([BOS, a, x, SEP, b, x, SEP, QRY, a, x])
    rec, frq = shortcut_hits(t2, [9])
    assert rec.tolist() == [True] and frq.tolist() == [True]


def test_loss_terms_full_plus_masked_over_micro_batches():
    import torch
    import torch.nn.functional as F
    from train0b import loss_terms
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 10, 7, generator=g, requires_grad=True)
    y = torch.randint(0, 7, (4, 10), generator=g)
    m = torch.rand(4, 10, generator=g) < 0.3
    m[0, 0] = True
    n_m, n_all = m.sum(), m.numel()
    ce_all = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    ce_m = F.cross_entropy(logits[m], y[m])
    # summing two micro-batches gives the batch means, for both loss kinds
    for kind, want in (("full_plus_masked", ce_all + ce_m), ("masked", ce_m)):
        parts = [loss_terms(logits[s:s + 2], y[s:s + 2], m[s:s + 2], n_m, n_all, kind) for s in (0, 2)]
        assert torch.allclose(sum(p[0] for p in parts), want, atol=1e-6)
        assert torch.allclose(sum(p[1] for p in parts), ce_m, atol=1e-6)
    # the full term puts gradient on unmasked positions; the masked loss does not
    for kind, grads_off_mask in (("full_plus_masked", True), ("masked", False)):
        logits.grad = None
        loss_terms(logits, y, m, n_m, n_all, kind)[0].backward()
        assert bool(logits.grad[~m].abs().sum() > 0) == grads_off_mask


def test_shortcut_values_at_end_match_hits():
    import numpy as np
    from eval0b import shortcut_hits, shortcut_values_at_end
    from sim0b.stream import K0, QRY, make_sequence
    s = make_sequence(np.random.default_rng(12), 1024, exc_rate=0.2, probes=False)
    t = s["tokens"]
    rec_v, frq_v = shortcut_values_at_end(t)
    # appending a probe answer for each value: the end-of-stream shortcut hits iff the answer equals it
    for v in {rec_v, frq_v, int(t[-1])}:
        t2 = np.concatenate([t, [QRY, K0, K0 + 1, v]])                 # QRY k_a k_b v: probe values are not facts
        r, f = shortcut_hits(t2, [len(t2) - 1])
        assert r[0] == (v == rec_v) and f[0] == (v == frq_v)
