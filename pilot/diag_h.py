"""Diagnostic (validation seeds only; reads labels, so it is never an arm): is the hippocampus failing at
selection (which tokens are stored) or at reading (retrieval and the rotated-key insertion)?

1. Coverage: fraction of fact-sentence tokens and of value tokens that the NLL rule keeps.
2. Oracle store: H keeps exactly the fact-sentence tokens (labels != 0), same capacity and read path.
"""
import sys

import numpy as np

from pilot.engine import Engine
from pilot.memory import EXTRA, Hippocampus
from pilot.run_pilot import STREAM, W, metrics
from pilot.text_stream import load_filler, make_stream


class Oracle(Hippocampus):
    labels = None

    def select(self, pos, nll):
        return pos[self.labels[pos] != 0]


def main(model, m, n_read):
    eng = Engine(model)
    filler = load_filler(eng.tok)
    for seed in (0, 1):
        s = make_stream(np.random.default_rng([seed, 500]), eng.tok, filler, **dict(STREAM, exc_rate=0.5))
        h = Hippocampus(eng, cap=EXTRA, m=m, n_read=n_read)
        out = eng.run(s["tokens"], W, memory=h, probes=s["probes"])
        nll = out["nll"]
        sel = h.pos
        lab = s["labels"]
        vals = np.zeros(len(lab), bool)
        for key, val, pos, kind in s["facts"]:
            vals[pos:pos + len(eng.tok(val, add_special_tokens=False)["input_ids"])] = True
        fact = lab != 0
        print(f"seed {seed} m {m}: kept {len(sel)} tokens; fact tokens kept {np.isin(np.flatnonzero(fact), sel).mean():.2f}, "
              f"value tokens kept {np.isin(np.flatnonzero(vals), sel).mean():.2f}, kept that are fact {fact[sel].mean():.2f}")
        mt = metrics(s["probes"], out["probes"])
        print(f"   nll-H: exc_beyond {mt['exc_acc_beyond']:.3f} rule {mt['rule_acc']:.3f}")
        Oracle.labels = lab
        o = Oracle(eng, cap=EXTRA, n_read=n_read)
        mt = metrics(s["probes"], eng.run(s["tokens"], W, memory=o, probes=s["probes"])["probes"])
        print(f"   oracle-H (stores {o.writes} fact tokens): exc_beyond {mt['exc_acc_beyond']:.3f} rule {mt['rule_acc']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
