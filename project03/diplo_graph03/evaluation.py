"""
Avaliação por ENTIDADE (não por subtoken): uma entidade só conta como acerto se tipo e
limites (início e fim) forem exatamente os mesmos da referência.
"""

import numpy as np


def bio_spans(tags) -> list[tuple]:
    """Converte uma sequência BIO em entidades (tipo, início, fim exclusivo)."""
    spans, typ, start = [], None, 0
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                spans.append((typ, start, i))
                typ = None
            if t != "O":
                typ, start = t[2:], i
    return spans


def entity_f1(y_true, y_pred, doc) -> dict:
    """
    y_true, y_pred: tags BIO por subtoken; doc: id do caso de cada subtoken (mesma ordem).
    Retorna {tipo: {p, r, f1, support}, "micro": {...}}.
    """
    y_true, y_pred, doc = map(np.asarray, (y_true, y_pred, doc))
    counts = {}   # tipo -> [tp, fp, fn]

    bounds = np.flatnonzero(doc[1:] != doc[:-1]) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(doc)]):
        gold = set(bio_spans(y_true[s:e]))
        pred = set(bio_spans(y_pred[s:e]))
        for typ, *_ in gold | pred:
            counts.setdefault(typ, [0, 0, 0])
        for sp in gold & pred:
            counts[sp[0]][0] += 1
        for sp in pred - gold:
            counts[sp[0]][1] += 1
        for sp in gold - pred:
            counts[sp[0]][2] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}

    report = {t: prf(*c) for t, c in sorted(counts.items())}
    report["micro"] = prf(*(sum(c[i] for c in counts.values()) for i in range(3)))
    return report


def print_report(report: dict) -> None:
    print(f"{'tipo':28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>6s}")
    for typ, m in report.items():
        print(f"{typ:28s} {m['p']:6.3f} {m['r']:6.3f} {m['f1']:6.3f} {m['support']:6d}")
