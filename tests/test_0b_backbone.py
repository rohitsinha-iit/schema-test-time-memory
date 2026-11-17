"""Stage 0b backbone: chunked SWA, gist update and chunk-level causality."""
import pytest

torch = pytest.importorskip("torch")

from model0b.backbone import Backbone, chunked_swa, dense_swa_reference  # noqa: E402
from model0b.gist import block_update, naive_update, read_states  # noqa: E402

DT = torch.float64


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


@pytest.mark.parametrize("window", [8, 16, 32])
def test_chunked_swa_equals_dense_reference(window):
    B, H, T, d, c = 2, 3, 64, 8, 8
    q, k, v = _rand(B, H, T, d, seed=1), _rand(B, H, T, d, seed=2), _rand(B, H, T, d, seed=3)
    assert torch.allclose(chunked_swa(q, k, v, c, window), dense_swa_reference(q, k, v, c, window), atol=1e-12)


def _gist_inputs(B=2, H=3, T=48, d=8, seed=0, decay=1.0):
    K = torch.nn.functional.normalize(_rand(B, H, T, d, seed=seed), dim=-1)
    V = _rand(B, H, T, d, seed=seed + 1)
    la = -decay * torch.nn.functional.softplus(_rand(B, H, T, seed=seed + 2))
    beta = torch.sigmoid(_rand(B, H, T, seed=seed + 3))
    return K, V, la, beta


def test_gist_block_update_equals_naive_loop():
    K, V, la, beta = _gist_inputs(T=16)
    S0 = _rand(2, 3, 8, 8, seed=9)
    assert torch.allclose(block_update(S0, K, V, la, beta), naive_update(S0, K, V, la, beta), atol=1e-10)
    # chained over blocks, as used by read_states
    a = read_states(K, V, la, beta, chunk=4, window=8, update=block_update)
    b = read_states(K, V, la, beta, chunk=4, window=8, update=naive_update)
    assert torch.allclose(a, b, atol=1e-10)


def test_gist_reads_only_blocks_evicted_before_the_chunk():
    c, W = 4, 8
    K, V, la, beta = _gist_inputs(T=48, decay=0.01)               # slow decay: old blocks stay measurable
    base = read_states(K, V, la, beta, c, W)
    n, lag = 48 // c, W // c
    for blk in range(n):
        V2 = V.clone(); V2[:, :, blk * c:(blk + 1) * c] += 1.0
        S = read_states(K, V2, la, beta, c, W)
        changed = [(S[:, :, j] - base[:, :, j]).abs().max().item() > 1e-9 for j in range(n + 1)]
        # block blk leaves the window after chunk blk + lag; only chunks after that may see it
        assert changed == [j >= blk + lag + 1 for j in range(n + 1)]


def _model():
    torch.manual_seed(0)
    return Backbone(vocab=50, d_model=32, n_layers=2, n_heads=2, chunk=8).to(DT).eval()


def test_model_no_position_reads_later_tokens():
    m, W = _model(), 16
    tok = torch.randint(4, 50, (2, 64), generator=torch.Generator().manual_seed(1))
    base = m(tok, W)
    for p in [5, 8, 23, 40, 63]:
        t2 = tok.clone(); t2[:, p] = (t2[:, p] + 7) % 46 + 4
        out = m(t2, W)
        assert torch.allclose(out[:, :p], base[:, :p], atol=1e-12)
        assert not torch.allclose(out[:, p:], base[:, p:])


def test_gist_carries_information_beyond_the_window():
    torch.manual_seed(0)
    m, W, c = Backbone(vocab=50, d_model=32, n_layers=1, n_heads=2, chunk=8).to(DT).eval(), 16, 8
    tok = torch.randint(4, 50, (1, 64), generator=torch.Generator().manual_seed(2))
    t2 = tok.clone(); t2[:, 3] = (t2[:, 3] + 7) % 46 + 4             # block 0, leaves after chunk W/c = 2
    diff = lambda: (m(t2, W) - m(tok, W)).abs().amax(-1)[0]
    d = diff()
    # chunks 1-2 still see position 3 in the window; from chunk 3 on only the gist can carry it
    assert d[c:3 * c].min() > 0 and d[3 * c:].min() > 0
    torch.nn.init.zeros_(m.blocks[0].gist_o.weight)
    d = diff()
    assert d[c:3 * c].min() > 0 and d[3 * c:].max() == 0


def test_probe_path_equals_forward_on_appended_chunk():
    m, W, c = _model(), 16, 8
    tok = torch.randint(4, 50, (2, 64), generator=torch.Generator().manual_seed(3))
    probes = torch.randint(4, 50, (2, 5, 3), generator=torch.Generator().manual_seed(4))
    logits, states = m(tok, W, return_state=True)
    pl = m.probe_logits(probes, states)
    for p in range(5):
        ext = torch.cat([tok, probes[:, p], torch.zeros(2, c - 3, dtype=tok.dtype) + 4], dim=1)
        ref = m(ext, W)[:, 64:67]
        assert torch.allclose(pl[:, p], ref, atol=1e-10)


def test_dense_no_gist_equals_chunked_with_full_window():
    # with W >= T nothing leaves the window, so the gist reads zero and chunked SWA is full causal attention
    torch.manual_seed(0)
    a = Backbone(vocab=50, d_model=32, n_layers=2, n_heads=2, chunk=8).to(DT).eval()
    b = Backbone(vocab=50, d_model=32, n_layers=2, n_heads=2, chunk=8, dense=True, gist=False).to(DT).eval()
    b.load_state_dict(a.state_dict())
    tok = torch.randint(4, 50, (2, 64), generator=torch.Generator().manual_seed(5))
    assert torch.allclose(a(tok, 64), b(tok, 64), atol=1e-12)


# ---- spec section 12, item 4: tests that must pass before any training ----

def _models_for_causality():
    from model0b.reference import GPT
    torch.manual_seed(0)
    return {
        "chunked_swa_gist": Backbone(vocab=50, d_model=32, n_layers=2, n_heads=2, chunk=8).to(DT).eval(),
        "dense_no_gist": Backbone(vocab=50, d_model=32, n_layers=2, n_heads=2, chunk=8, dense=True, gist=False).to(DT).eval(),
        "reference_gpt": GPT(vocab=50, d_model=32, n_layers=2, n_heads=2, max_len=64).to(DT).eval(),
    }


@pytest.mark.parametrize("name", ["chunked_swa_gist", "dense_no_gist", "reference_gpt"])
def test_causality_changing_all_later_tokens(name):
    m = _models_for_causality()[name]
    g = torch.Generator().manual_seed(6)
    tok = torch.randint(4, 50, (2, 64), generator=g)
    base = m(tok, 16)
    for i in [0, 7, 8, 31, 62]:
        t2 = tok.clone(); t2[:, i + 1:] = torch.randint(4, 50, (2, 63 - i), generator=g)
        out = m(t2, 16)
        assert torch.allclose(out[:, :i + 1], base[:, :i + 1], atol=1e-12), (name, i)
        assert not torch.allclose(out[:, i + 1:], base[:, i + 1:])


def test_rope_scores_invariant_to_position_shift():
    from model0b.backbone import rope
    q, k = _rand(1, 2, 32, 64, seed=7), _rand(1, 2, 32, 64, seed=8)
    pos = torch.arange(32)
    s0 = rope(q, pos) @ rope(k, pos).transpose(-1, -2)
    for shift in (1, 37, 1000):
        s1 = rope(q, pos + shift) @ rope(k, pos + shift).transpose(-1, -2)
        assert torch.allclose(s0, s1, atol=1e-3, rtol=1e-4), shift        # angles are float32 inside rope
    s_bad = rope(q, pos) @ rope(k, pos + 1).transpose(-1, -2)              # a relative shift must change them
    assert not torch.allclose(s0, s_bad, atol=1e-2)


@pytest.mark.parametrize("dense", [True, False])
def test_one_layer_learns_previous_token(dense):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = Backbone(vocab=32, d_model=64, n_layers=1, n_heads=2, chunk=16, dense=dense, gist=not dense).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(500):
        tok = torch.randint(0, 32, (32, 64), generator=g, device=dev)
        loss = torch.nn.functional.cross_entropy(m(tok, 32)[:, 1:].reshape(-1, 32), tok[:, :-1].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tok = torch.randint(0, 32, (64, 64), generator=g, device=dev)
        acc = (m(tok, 32)[:, 1:].argmax(-1) == tok[:, :-1]).float().mean().item()
    assert acc > 0.99, acc
