"""Stage 0b evaluation. G0 (stage0b_spec.md, section 7): within-window probe accuracy and beyond-window
exception accuracy of the backbone (+ gist) at a given attention window.

Evaluation streams have no interleaved probes. After the stream, every recurring rule and every exception
is asked as an independent probe `QRY k_a k_b -> v` that reads the window cache and the gist state at the
end of the stream; nothing is written. Delay = tokens from the key's last occurrence (k_a) to the end of
the stream. Buckets: within window (delay <= W), W < delay <= 4W, delay > 4W.
"""
import numpy as np
import torch

from sim0b.stream import F0, K0, QRY, V0, make_sequence


def _shortcut_scan(tokens, positions):
    """Shared scan for shortcut_hits: the (most recent, most frequent) fact value before each position in
    `positions`, where position len(tokens) means "after the stream". Ties in frequency: the most recent."""
    targets = set(int(p) for p in positions)
    counts, last_seen, recent, out = {}, {}, None, {}
    for p in range(len(tokens) + 1):
        if p in targets:
            out[p] = recent, (max(counts, key=lambda v: (counts[v], last_seen[v])) if counts else None)
        if p == len(tokens):
            break
        t = tokens[p]
        if V0 <= t < F0 and p >= 2 and K0 <= tokens[p - 1] < V0:
            j = p - 1
            while j > 0 and K0 <= tokens[j - 1] < V0:
                j -= 1
            if tokens[j - 1] != QRY:
                counts[t] = counts.get(t, 0) + 1; last_seen[t] = p; recent = t
    return out


def shortcut_hits(tokens, positions):
    """Key-free shortcut predictions for the targets at `positions` (spec sections 11 and 12): the value of the
    most recent fact before it, and the most frequent fact value so far (ties: the most recent). A fact value is
    a value token after a run of key tokens not preceded by QRY. Reads tokens only, never labels."""
    out = _shortcut_scan(tokens, positions)
    pos = [int(p) for p in positions]
    return (np.array([out[p][0] == tokens[p] for p in pos], bool),
            np.array([out[p][1] == tokens[p] for p in pos], bool))


def shortcut_values_at_end(tokens):
    """(most recent, most frequent) fact value after the whole stream: the shortcut answers to every eval probe."""
    return _shortcut_scan(tokens, [len(tokens)])[len(tokens)]


RANK_BUCKETS = ((1, 4), (5, 8), (9, 16))     # rule frequency rank within the stream (1 = most frequent)


def _acc(c):
    return float(np.mean(c)) if len(c) else float("nan")


@torch.no_grad()
def g0_eval(model, window, exc_rate, n_streams=64, n_tokens=4096, eval_seed=100, batch=16, corr=0.3, device="cuda"):
    rng = np.random.default_rng([eval_seed, int(round(exc_rate * 1000))])
    seqs = [make_sequence(rng, n_tokens, exc_rate=exc_rate, corr=corr, probes=False) for _ in range(n_streams)]
    rows = []                                   # (kind, delay, correct, rank, recent shortcut hit, freq shortcut hit)
    model.eval()
    for i in range(0, n_streams, batch):
        part = seqs[i:i + batch]
        tok = torch.as_tensor(np.stack([s["tokens"] for s in part]), device=device)
        P = max(len(s["probes"]) for s in part)
        pt = np.full((len(part), P, 3), QRY, np.int64)
        for b, s in enumerate(part):
            for p, pr in enumerate(s["probes"]):
                pt[b, p, 1:] = pr["ka"], pr["kb"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, states = model(tok, window, return_state=True)
            pred = model.probe_logits(torch.as_tensor(pt, device=device), states)[:, :, 2].argmax(-1).cpu().numpy()
        for b, s in enumerate(part):
            rec_v, frq_v = shortcut_values_at_end(s["tokens"])
            rules = sorted((pr for pr in s["probes"] if pr["kind"] == "rule"), key=lambda pr: -pr["count"])
            rank = {(pr["ka"], pr["kb"]): i + 1 for i, pr in enumerate(rules)}   # ties: generator order
            for p, pr in enumerate(s["probes"]):
                rows.append((pr["kind"], pr["delay"], pred[b, p] == pr["v"], rank.get((pr["ka"], pr["kb"]), 0),
                             rec_v == pr["v"], frq_v == pr["v"]))
    kind, d, c, rk, sr, sf = (np.array([r[i] for r in rows]) for i in range(6))
    exc, rule, win = kind == "exc", kind == "rule", d <= window
    by_rank = {}
    for lo, hi in RANK_BUCKETS:
        m = rule & (rk >= lo) & (rk <= hi)
        by_rank.update({f"rule_acc_rank{lo}to{hi}": _acc(c[m]), f"shortcut_freq_rule_rank{lo}to{hi}": _acc(sf[m]),
                        f"n_rule_rank{lo}to{hi}": int(m.sum())})
    return dict(
        shortcut_recent_rule=_acc(sr[rule]), shortcut_freq_rule=_acc(sf[rule]), shortcut_recent_exc=_acc(sr[exc]),
        shortcut_freq_exc=_acc(sf[exc]), shortcut_recent_within=_acc(sr[win]), shortcut_freq_within=_acc(sf[win]),
        **by_rank,
        within_acc=_acc(c[win]), within_rule_acc=_acc(c[win & rule]), within_exc_acc=_acc(c[win & exc]),
        exc_beyond_acc=_acc(c[exc & ~win]), exc_1to4W_acc=_acc(c[exc & ~win & (d <= 4 * window)]),
        exc_gt4W_acc=_acc(c[exc & (d > 4 * window)]), rule_acc=_acc(c[rule]), rule_beyond_acc=_acc(c[rule & ~win]),
        n_within=int(win.sum()), n_exc_beyond=int((exc & ~win).sum()), n_exc_1to4W=int((exc & ~win & (d <= 4 * window)).sum()),
        n_exc_gt4W=int((exc & (d > 4 * window)).sum()), n_rule=int(rule.sum()))
