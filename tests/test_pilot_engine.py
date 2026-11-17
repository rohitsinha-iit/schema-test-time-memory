"""Pilot engine: chunked full attention equals the model's own forward; RoPE inverse; window causality."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from pilot.engine import Engine  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    try:
        return Engine(dtype=torch.float32)
    except Exception as e:
        pytest.skip(str(e))


def test_chunked_full_attention_matches_model(eng):
    from transformers import AutoModelForCausalLM
    rng = np.random.default_rng(0)
    toks = rng.integers(1000, 20000, 300)
    ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32,
                                               attn_implementation="eager").cuda().eval()
    with torch.no_grad():
        lp = torch.log_softmax(ref(torch.as_tensor(toks, device="cuda")[None]).logits[0].float(), -1)
    want = -lp[:-1].gather(-1, torch.as_tensor(toks[1:], device="cuda")[:, None])[:, 0].cpu().numpy()
    got = eng.run(toks, W=10**9, chunk=64)["nll"][1:]
    assert np.abs(got - want).max() < 1e-3


def test_rope_inverse(eng):
    x = torch.randn(2, 5, eng.d, device="cuda")
    pos = np.arange(100, 105)
    assert torch.allclose(eng.rotate(eng.rotate(x, pos), pos, inverse=True), x, atol=1e-4)


def test_causal_and_window_changes_output(eng):
    rng = np.random.default_rng(1)
    a = rng.integers(1000, 20000, 400)
    b = a.copy(); b[200:] = rng.integers(1000, 20000, 200)
    na, nb = eng.run(a, W=128)["nll"], eng.run(b, W=128)["nll"]
    assert np.abs(na[:200] - nb[:200]).max() < 1e-4                 # later tokens never change earlier outputs
    full = eng.run(a, W=10**9)["nll"]
    assert np.abs(na[:132] - full[:132]).max() < 1e-4                # identical while everything fits the window
    assert np.abs(na[300:] - full[300:]).max() > 1e-2                # the window matters later
