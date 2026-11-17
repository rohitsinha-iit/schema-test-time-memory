"""Pure-torch gated delta rule with per-token outputs equals the per-token recurrence."""
import pytest

torch = pytest.importorskip("torch")

from pilot.gdn import block_outputs, naive_outputs  # noqa: E402


def test_block_outputs_match_naive():
    g = torch.Generator().manual_seed(0)
    B, H, c, d = 2, 3, 16, 8
    nrm = lambda x: torch.nn.functional.normalize(x, dim=-1)
    S0 = torch.randn(B, H, d, d, generator=g, dtype=torch.float64) * 0.3
    K, Q = nrm(torch.randn(B, H, c, d, generator=g, dtype=torch.float64)), nrm(torch.randn(B, H, c, d, generator=g, dtype=torch.float64))
    V = torch.randn(B, H, c, d, generator=g, dtype=torch.float64)
    la = -torch.rand(B, H, c, generator=g, dtype=torch.float64) * 0.5
    be = torch.rand(B, H, c, generator=g, dtype=torch.float64)
    be[..., 3] = 0.0                                                   # a "no write" token
    O1, S1 = naive_outputs(S0, K, V, Q, la, be, 0.5)
    O2, S2 = block_outputs(S0, K, V, Q, la, be, 0.5)
    assert torch.allclose(O1, O2, atol=1e-10) and torch.allclose(S1, S2, atol=1e-10)
