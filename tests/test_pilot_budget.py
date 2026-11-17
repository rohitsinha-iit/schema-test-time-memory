"""Pilot arms hold equal memory: window + H + C + A pairs per layer per KV head (hard rule 2)."""
import pytest

pytest.importorskip("torch")

from pilot.memory import EXTRA as H_ONLY_EXTRA  # noqa: E402
from pilot.schema_memory import EXTRA, SPECS, mem_pairs  # noqa: E402


def test_equal_memory_budget_pilot():
    W = 1024
    assert H_ONLY_EXTRA == EXTRA                                   # hippo arm and swa_big use the same extra pairs
    for name, (extra_w, kw) in SPECS.items():
        assert extra_w + mem_pairs(kw) == EXTRA, name


def test_equal_memory_budget_linear_cortex():
    from pilot.linear_cortex import EXTRA as LIN_EXTRA, LIN_SPECS, lin_pairs
    assert LIN_EXTRA == EXTRA
    for name, (extra_w, kw) in LIN_SPECS.items():
        if not name.startswith("diag_"):
            assert extra_w + lin_pairs(kw) == EXTRA, name


def test_equal_memory_budget_lora_cortex():
    from pilot.lora_cortex import CFG_15B, EXTRA as L_EXTRA, LORA_P, LORA_SPECS, RANK, lora_pairs, spec_pairs
    assert L_EXTRA == EXTRA and LORA_P >= lora_pairs(CFG_15B, RANK)
    for name, (extra_w, kw) in LORA_SPECS.items():
        assert extra_w + spec_pairs(kw) == EXTRA, name


def test_confirmatory_arms_match_uniform_bytes():
    from pilot.lora_cortex import LORA_SPECS
    hu = LORA_SPECS["hrd_uniform"][1]["H_cap"]
    for name in ("hrd_schema_ng", "hrd_delta_pri", "hrd_reuse_pri"):
        kw = LORA_SPECS[name][1]
        assert kw["H_cap"] == hu and kw["gate"] is False and kw["R"] == 0, name
    assert LORA_SPECS["hrd_delta_pri"][1]["priority"] == "delta" and LORA_SPECS["hrd_reuse_pri"][1]["priority"] == "reuse"
