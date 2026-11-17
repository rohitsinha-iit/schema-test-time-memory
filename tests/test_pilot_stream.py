"""Pilot text stream: values in the token stream match the probes, exceptions are one-off, drift and correlation."""
import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from pilot.text_stream import ATTRS, load_filler, make_stream  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok_filler():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        filler = load_filler(tok)
    except Exception as e:                     # model or filler cache not on this machine
        pytest.skip(str(e))
    return tok, filler


def test_stream_values_and_probes(tok_filler):
    tok, filler = tok_filler
    s = make_stream(np.random.default_rng(0), tok, filler, n_tokens=4096, exc_rate=0.3)
    t = s["tokens"]
    assert len(t) == len(s["labels"]) == 4096
    for key, val, pos, kind in s["facts"]:                  # the value tokens sit at the recorded position
        v = tok(val, add_special_tokens=False)["input_ids"]
        assert t[pos:pos + len(v)].tolist() == v
        assert s["labels"][pos] == (1 if kind == "rule" else 2)
    last = {}
    for key, val, pos, kind in s["facts"]:
        last[key] = (val, pos)
    for pr in s["probes"]:
        attr = [i for i, (a, _) in enumerate(ATTRS) if a == pr["attr"]][0]
        val, pos = last[(pr["person"], attr)]
        assert pr["value"] == val and pr["delay"] == 4096 - pos
        assert tok.decode(pr["target"]).strip() == val
        assert pr["person"] in tok.decode(pr["prompt"])


def test_exceptions_one_off_rules_recur_drift_and_corr(tok_filler):
    tok, filler = tok_filler
    s = make_stream(np.random.default_rng(1), tok, filler, n_tokens=8192, exc_rate=0.5, corr=1.0, drift_frac=0.5)
    exc = [k for k, v, p, kind in s["facts"] if kind == "exc"]
    rules = {k for k, v, p, kind in s["facts"] if kind == "rule"}
    assert len(exc) == len(set(exc)) and not set(exc) & rules
    assert all(k[0] in {r[0] for r in rules} | {e[0] for e in exc} for k in exc)
    rule_people = {k[0] for k in rules}
    assert np.mean([k[0] in rule_people for k in exc]) > 0.9           # corr 1: exceptions reuse rule persons
    assert all(p["count"] >= 2 for p in s["probes"] if p["kind"] == "rule")
    dr = [p for p in s["probes"] if p["drifted"] and p["old_value"] is not None]
    assert dr and all(p["old_value"] != p["value"] for p in dr)
    assert (s["labels"] == 0).mean() > 0.5                              # most tokens are filler


def test_stream_identical_across_processes():
    """Regression (4 Nov 2026): names were drawn into a set of str and listed in hash order, so the same seed gave
    a different stream in every process (PYTHONHASHSEED). Two processes with different hash seeds must agree."""
    import os
    import subprocess
    import sys
    code = ("import numpy as np, hashlib\n"
            "from transformers import AutoTokenizer\n"
            "from pilot.text_stream import load_filler, make_stream\n"
            "tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True)\n"
            "s = make_stream(np.random.default_rng(3), tok, load_filler(tok), n_tokens=2048)\n"
            "print(hashlib.sha256(s['tokens'].tobytes()).hexdigest())\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = [subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": root}).stdout.strip().splitlines()[-1:]
           for h in ("1", "2")]
    assert out[0] and out[0] == out[1]
