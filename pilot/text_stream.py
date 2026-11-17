"""One-day pilot stream (stage0b_v2_spec.md section 2, cut down): facts about invented people, written as sentences
between distractor passages of natural text (wikitext-103).

* A key is (person, attribute); a fact sentence states its value with one of 3 paraphrase templates.
* Rules: n_rules keys that recur with Zipf frequencies. Exceptions: one-off keys, with probability exc_rate per fact.
  With probability corr an exception reuses a rule's person with a different attribute (key correlation).
* Drift: at the middle of the stream, drift_frac of the rules get new values.
* Filler: a contiguous span of filler_min..filler_max wikitext tokens before every fact.
* Evaluation probes (read-only, after the stream): "Question: What is the A of P?\nAnswer: The A of P is" -> " V".
  One per rule seen at least twice and one per exception. Delay = tokens from the last value to the end.
Labels (0 filler, 1 rule fact, 2 exception fact) and probe kinds are for evaluation only; memory never reads them.
"""
import os

import numpy as np

CONS = list("bdfgklmnprstvz") + ["sh", "th", "br", "dr", "kr", "tr"]
VOWS = list("aeiou") + ["ai", "ou", "ei"]
ATTRS = [("vault code", "num"), ("badge number", "num"), ("locker number", "num"), ("room number", "num"),
         ("parking spot", "num"), ("ticket number", "num"), ("hometown", "word"), ("pet's name", "word"),
         ("boat's name", "word"), ("favorite dish", "word")]
TEMPLATES = ["The {a} of {p} is {v}.", "{p}'s {a} is {v}.", "For {p}, the {a} is {v}."]
PROBE = "\nQuestion: What is the {a} of {p}?\nAnswer: The {a} of {p} is"
FILLER_NPY = os.path.join(os.path.dirname(__file__), "data", "filler_ids.npy")


def _word(rng, n_syl):
    w = "".join(CONS[rng.integers(len(CONS))] + VOWS[rng.integers(len(VOWS))] for _ in range(n_syl))
    if rng.random() < 0.5:
        w += "nrlsk"[rng.integers(5)]
    return w.capitalize()


def _value(rng, kind):
    return str(int(rng.integers(1000, 10000))) if kind == "num" else _word(rng, 2)


def load_filler(tok, parquet=None):
    """wikitext-103 token ids as one array (cached in pilot/data). Lines that are headings or empty are dropped."""
    if os.path.exists(FILLER_NPY):
        return np.load(FILLER_NPY)
    import pandas as pd
    lines = [t for t in pd.read_parquet(parquet)["text"] if len(t) > 200 and not t.strip().startswith("=")]
    ids = np.array(tok("".join(lines))["input_ids"], np.int32)
    os.makedirs(os.path.dirname(FILLER_NPY), exist_ok=True)
    np.save(FILLER_NPY, ids)
    return ids


def make_stream(rng, tok, filler, n_tokens=8192, n_rules=16, zipf=1.0, exc_rate=0.2, corr=0.5, drift_frac=0.25,
                filler_min=20, filler_max=80):
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    people = set()
    while len(people) < 4 * n_rules:
        people.add(f"{_word(rng, 2)} {_word(rng, 2)}")
    people = sorted(people)                          # sets of str iterate in a per-process hash order
    rule_keys = [(people[i], int(rng.integers(len(ATTRS)))) for i in range(n_rules)]
    rule_val = [_value(rng, ATTRS[a][1]) for _, a in rule_keys]
    freq = 1.0 / np.arange(1, n_rules + 1) ** zipf
    freq /= freq.sum()
    drift_ids = set(rng.choice(n_rules, int(round(drift_frac * n_rules)), replace=False).tolist())
    used, next_person = set(rule_keys), n_rules
    toks, labels, facts, drifted = [], [], [], False      # facts: (key, value, value_pos, kind, rule id)

    while True:
        if not drifted and len(toks) >= n_tokens // 2:
            for r in drift_ids:
                rule_val[r] = _value(rng, ATTRS[rule_keys[r][1]][1])
            drifted = True
        if rng.random() < exc_rate:
            while True:
                if rng.random() < corr:
                    p = rule_keys[rng.integers(n_rules)][0]
                else:
                    p, next_person = people[next_person % len(people)], next_person + 1
                key = (p, int(rng.integers(len(ATTRS))))
                if key not in used:
                    break
            used.add(key)
            val, kind, rid = _value(rng, ATTRS[key[1]][1]), "exc", -1
        else:
            rid = int(rng.choice(n_rules, p=freq))
            key, val, kind = rule_keys[rid], rule_val[rid], "rule"
        a, p = ATTRS[key[1]][0], key[0]
        s = TEMPLATES[rng.integers(len(TEMPLATES))].format(a=a, p=p, v="\x00")
        pre, post = s.split("\x00")
        f0 = int(rng.integers(len(filler) - filler_max))
        fill = filler[f0:f0 + int(rng.integers(filler_min, filler_max + 1))].tolist()
        piece_pre, piece_val, piece_post = enc("\n" + pre), enc(val), enc(post + "\n")
        if len(toks) + len(fill) + len(piece_pre) + len(piece_val) + len(piece_post) > n_tokens:
            break
        toks += fill; labels += [0] * len(fill)
        lab = 1 if kind == "rule" else 2
        toks += piece_pre; labels += [lab] * len(piece_pre)
        facts.append((key, val, len(toks), kind, rid))
        toks += piece_val + piece_post; labels += [lab] * (len(piece_val) + len(piece_post))
    f0 = int(rng.integers(len(filler) - n_tokens))
    pad = n_tokens - len(toks)
    toks += filler[f0:f0 + pad].tolist(); labels += [0] * pad

    last, count = {}, {}
    for i, (key, *_r) in enumerate(facts):
        last[key] = i; count[key] = count.get(key, 0) + 1
    probes = []
    for key, i in last.items():
        _, val, vpos, kind, rid = facts[i]
        if kind == "rule" and count[key] < 2:
            continue
        old = None
        if kind == "rule" and rid in drift_ids:
            old = next((f[1] for f in reversed(facts) if f[0] == key and f[2] < n_tokens // 2 and f[1] != val), None)
        a = ATTRS[key[1]][0]
        probes.append(dict(person=key[0], attr=a, value=val, kind=kind, delay=n_tokens - vpos, count=count[key],
                           drifted=kind == "rule" and rid in drift_ids, old_value=old,
                           prompt=enc(PROBE.format(a=a, p=key[0])), target=enc(" " + val)))
    return dict(tokens=np.array(toks, np.int64), labels=np.array(labels, np.int8), probes=probes,
                facts=[(k, v, pos, kind) for k, v, pos, kind, _ in facts])
