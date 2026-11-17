"""Soft-slot cortex interference diagnostic (validation seeds; labels used only to split landmarks, never an arm).

Cortex: C_cap slots (key, value) per layer and KV head. Write (k, v): if the best slot cosine is below theta_alloc
the item takes a free slot (or the least used one); otherwise the values of its top-k slots move by the delta rule
V_s += eta w_s (v - y), y = sum_s w_s V_s, w = softmax(cos / tau) over the top-k. Read: y(k) = sum over the top-k.
k = 1 is the pilot's hard slot cortex. Larger k shares slots between items, so writes can interfere.

Protocol on the real landmarks of a stream (pre-RoPE keys, values; selector-chosen tokens):
  1. write every recurring-rule landmark in stream order; measure the rules' read error e0;
  2. write every exception landmark; measure the rules' read error e1 (damage = e1 - e0) and the exceptions' error.
Errors are ||y - v||^2 / ||v||^2, averaged over the probed layers and KV heads.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pilot.engine import Engine
from pilot.memory import Hippocampus
from pilot.run_pilot import STREAM, W
from pilot.text_stream import load_filler, make_stream


class SoftSlots:
    def __init__(self, cap, d, k, tau=0.05, theta_alloc=0.9, eta=1.0, dev="cuda"):
        self.K, self.V = torch.zeros(cap, d, device=dev), torch.zeros(cap, d, device=dev)
        self.used, self.use = torch.zeros(cap, dtype=torch.bool, device=dev), torch.zeros(cap, device=dev)
        self.k, self.tau, self.theta, self.eta = k, tau, theta_alloc, eta

    def _w(self, x):
        cos = (F.normalize(x, dim=-1) @ F.normalize(self.K, dim=-1).T).masked_fill(~self.used, -2.0)
        c, i = cos.topk(min(self.k, len(self.used)), -1)
        w = torch.softmax(c / self.tau, -1) * (c > -2)
        return c, i, w

    def read(self, x):
        c, i, w = self._w(x)
        return (w[..., None] * self.V[i]).sum(-2)

    def write(self, x, v):
        for j in range(len(x)):
            c, i, w = self._w(x[j:j + 1])
            if not self.used.any() or c[0, 0] < self.theta:
                s = int(torch.nonzero(~self.used)[0]) if (~self.used).any() else int(self.use.argmin())
                self.K[s], self.V[s], self.used[s] = x[j], v[j], True
                self.use[s] += 1
            else:
                y = (w[0, :, None] * self.V[i[0]]).sum(0)
                self.V[i[0]] += self.eta * w[0, :, None] * (v[j] - y)
                self.use[i[0]] += w[0]


def err(mem, x, v):
    return float((((mem.read(x) - v) ** 2).sum(-1) / (v ** 2).sum(-1).clamp_min(1e-6)).mean())


def main():
    eng = Engine("Qwen/Qwen2.5-1.5B-Instruct")
    filler = load_filler(eng.tok)
    res = {}
    for seed in (0, 1):
        s = make_stream(np.random.default_rng([seed, 500]), eng.tok, filler, **dict(STREAM, exc_rate=0.5))
        h = Hippocampus(eng, cap=10**6)
        eng.run(s["tokens"], W, memory=h)
        lab = s["labels"][h.pos]
        rule, exc = torch.as_tensor(lab == 1, device=eng.dev), torch.as_tensor(lab == 2, device=eng.dev)
        for l in (7, 14, 21):
            for hd in range(eng.Hkv):
                K, V = h.k[l][hd].float(), h.v[l][hd].float()
                for k in (1, 2, 4, 8):
                    for th in (0.9, 0.98):
                        m = SoftSlots(768, eng.d, k, theta_alloc=th)
                        m.write(K[rule], V[rule])
                        e0 = err(m, K[rule], V[rule])
                        m.write(K[exc], V[exc])
                        e1, ex = err(m, K[rule], V[rule]), err(m, K[exc], V[exc])
                        res.setdefault((k, th), []).append((e0, e1, ex, int(m.used.sum())))
        print(f"seed {seed}: rule landmarks {int(rule.sum())}, exception landmarks {int(exc.sum())}", flush=True)
    for (k, th), v in sorted(res.items()):
        v = np.array(v)
        print(f"top-k {k} theta_alloc {th}: rule error after rules {v[:, 0].mean():.3f} -> after exceptions {v[:, 1].mean():.3f} "
              f"(damage {np.mean(v[:, 1] - v[:, 0]):+.3f}); exception error {v[:, 2].mean():.3f}; slots used {v[:, 3].mean():.0f}")


if __name__ == "__main__":
    main()
