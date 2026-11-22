
# código ainda NÃO testado

"""
Janelas: recorta cada caso em blocos de `window_lines` linhas do .txt.

Compartilhado pelas 3 abordagens (o seq2matrix usa janelas pequenas por causa de N e L; as outras
podem usar o mesmo recorte para caber no limite de tokens do encoder).

  1. Recorte        line_spans, window_spans, make_windows_from_text, make_windows
  2. Grafo por janela (ouro)   Window.graph / Window.local_graph()
  3. Fusão          merge_window_graphs (janelas -> grafo do caso)
  4. Estatísticas   window_stats, print_window_stats (definem max_nodes, max_len e o valor de window_lines)

Convenções:
  * window_lines = k >= 1: blocos SEM sobreposição de k linhas não vazias (o último pode ser menor).
    window_lines = 0 (ou None): o caso inteiro é uma janela só.
  * O grafo ouro da janela tem posições GLOBAIS (do caso inteiro); local_graph() dá as locais.
  * No grafo da janela, nodes[0] é o PATIENT (sem span), se o caso tem paciente; depois vêm os
    nós da janela na ordem do texto. Um nó com várias menções fica só com as que caem na janela.
  * Arestas semânticas só existem entre nós da MESMA janela (as que cruzam janelas se perdem,
    e window_stats conta quantas). Arestas do paciente existem para todo nó presente na janela.

CLI (estatísticas): python -m diplo_graph03.preprocessing.windows [--lines 1 2 3 0]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from diplo_graph03.schema import (
    PATIENT,
    PATIENT_EDGE_TYPES,
    Graph,
    GraphEdge,
    GraphNode,
)

# ================================================================ 1. Recorte

_LINE = re.compile(r"[^\r\n]*\S[^\r\n]*")      # uma linha com pelo menos um caractere visível


def line_spans(text: str) -> list[tuple[int, int]]:
    """Posições (início, fim) das linhas não vazias, sem os terminadores de linha."""
    return [(m.start(), m.end()) for m in _LINE.finditer(text)]


def window_spans(text: str, window_lines: int | None = 1) -> list[tuple[int, int]]:
    """Posições (início, fim) das janelas: blocos de `window_lines` linhas consecutivas."""
    lines = line_spans(text)
    if not lines:
        return []
    if not window_lines or window_lines <= 0:
        return [(lines[0][0], lines[-1][1])]
    return [(lines[i][0], lines[min(i + window_lines, len(lines)) - 1][1])
            for i in range(0, len(lines), window_lines)]


@dataclass
class Window:
    doc_id: str
    index: int                  # número da janela dentro do caso
    start: int                  # posição (em caracteres) no texto do caso
    end: int
    text: str                   # texto do caso entre start e end
    graph: Graph | None = None  # grafo ouro, com posições globais (None na inferência)

    def local_graph(self) -> Graph | None:
        """Grafo ouro com posições relativas ao início da janela (as do tokenizer do modelo)."""
        return None if self.graph is None else shift_graph(self.graph, -self.start)


def make_windows_from_text(doc_id: str, text: str, window_lines: int | None = 1) -> list[Window]:
    """Janelas sem grafo ouro (para inferência em texto novo)."""
    return [Window(doc_id, i, s, e, text[s:e])
            for i, (s, e) in enumerate(window_spans(text, window_lines))]


def make_windows(doc, window_lines: int | None = 1) -> list[Window]:
    """
    Janelas de um caso COM grafo ouro. `doc` é qualquer objeto com `id`, `text`, `entities` e
    `relations` já no esquema do grafo (saída do dataset.convert_folder).
    """
    windows = make_windows_from_text(doc.id, doc.text, window_lines)
    case = Graph.from_brat(doc.entities, doc.relations, doc.id)
    for w in windows:
        w.graph = window_graph(case, w.start, w.end, doc.text)
    return windows


# ================================================================ 2. Grafo por janela


def shift_graph(g: Graph, delta: int) -> Graph:
    """Cópia do grafo com todas as posições deslocadas por `delta` caracteres."""
    nodes = [GraphNode(n.type, [(s + delta, e + delta) for s, e in n.spans], n.text)
             for n in g.nodes]
    edges = [GraphEdge(e.type, e.src, e.dst) for e in g.edges]
    return Graph(nodes, edges, g.doc_id)


def window_graph(case: Graph, start: int, end: int, doc_text: str) -> Graph:
    """Recorta o grafo do caso para a janela [start, end) (ver as convenções no topo)."""
    pidx = case.patient_index()

    cand = []                                    # (1ª posição na janela, índice no caso, spans)
    for i, n in enumerate(case.nodes):
        if i == pidx:
            continue
        inside = [(s, e) for s, e in n.spans if s >= start and e <= end]
        if inside:
            cand.append((min(s for s, _ in inside), i, inside))
    cand.sort()

    nodes, remap = [], {}
    if pidx is not None:
        nodes.append(GraphNode(PATIENT, [], ""))
        remap[pidx] = 0
    for _, i, inside in cand:
        remap[i] = len(nodes)
        texts = list(dict.fromkeys(doc_text[s:e] for s, e in inside))
        nodes.append(GraphNode(case.nodes[i].type, inside, " | ".join(texts)))

    edges = [GraphEdge(e.type, remap[e.src], remap[e.dst])
             for e in case.edges if e.src in remap and e.dst in remap]
    return Graph(nodes, edges, case.doc_id)


# ================================================================ 3. Fusão das janelas


def merge_window_graphs(windows: list[Window], local_graphs: list[Graph], doc_id: str = "") -> Graph:
    """
    Junta os grafos (com posições LOCAIS) previstos para as janelas de um caso num grafo do caso
    (posições globais): um só PATIENT, nós com mesmo tipo e mesmas posições viram um, e arestas
    repetidas são removidas. Nós de janelas diferentes NÃO são fundidos por texto; a avaliação
    compara no nível de menção.
    """
    nodes, edges = [], set()
    index = {}                                   # (tipo, posições) -> índice no grafo final
    patient = None
    for w, local in zip(windows, local_graphs):
        g = shift_graph(local, w.start)
        remap = {}
        for i, n in enumerate(g.nodes):
            if n.type == PATIENT:
                if patient is None:
                    patient = len(nodes)
                    nodes.append(GraphNode(PATIENT, [], ""))
                remap[i] = patient
                continue
            key = (n.type, tuple(n.spans))
            if key not in index:
                index[key] = len(nodes)
                nodes.append(GraphNode(n.type, list(n.spans), n.text))
            remap[i] = index[key]
        for e in g.edges:
            if e.src in remap and e.dst in remap:
                edges.add((e.type, remap[e.src], remap[e.dst]))
    return Graph(nodes, [GraphEdge(t, s, d) for t, s, d in sorted(edges)], doc_id)


# ================================================================ 4. Estatísticas


def _pct(values: list, p: float):
    if not values:
        return 0
    xs = sorted(values)
    return xs[min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))]


def _summary(values: list) -> dict:
    return {"p50": _pct(values, 50), "p90": _pct(values, 90), "p95": _pct(values, 95),
            "p99": _pct(values, 99), "max": max(values) if values else 0}


def window_stats(docs, window_lines: int | None = 1, token_counter=None, max_len: int = 512) -> dict:
    """
    Estatísticas das janelas de uma lista de casos (já no esquema do grafo).

    token_counter: função texto -> nº de tokens (opcional; ex.: len(tokenizer(text)["input_ids"])).
    Devolve um dict com:
      nodes            percentis de nós (sem o paciente) por janela -> escolha de max_nodes (p95)
      tokens           percentis de tokens por janela (None sem token_counter) -> max_len
      over_max_len     fração de janelas com mais de max_len tokens
      empty_windows    fração de janelas sem nenhum nó
      semantic_edges   (total, preservadas, fração): arestas semânticas com os dois nós na mesma janela
      patient_edges    (total, preservadas, fração): arestas do paciente cujo nó aparece em alguma janela
    """
    nodes_per, tokens_per = [], []
    sem_total = sem_kept = pat_total = pat_kept = 0

    for doc in docs:
        wins = make_windows(doc, window_lines)
        case = Graph.from_brat(doc.entities, doc.relations, doc.id)
        pidx = case.patient_index()

        member = {i: {w.index for w in wins if any(s >= w.start and e <= w.end for s, e in n.spans)}
                  for i, n in enumerate(case.nodes) if i != pidx}

        for w in wins:
            nodes_per.append(sum(1 for n in w.graph.nodes if n.type != PATIENT))
            if token_counter is not None:
                tokens_per.append(token_counter(w.text))

        for e in case.edges:
            if e.type in PATIENT_EDGE_TYPES:
                pat_total += 1
                pat_kept += bool(member.get(e.dst))
            else:
                sem_total += 1
                sem_kept += bool(member.get(e.src, set()) & member.get(e.dst, set()))

    frac = lambda a, b: a / b if b else 1.0
    return {
        "window_lines": window_lines,
        "n_windows": len(nodes_per),
        "nodes": _summary(nodes_per),
        "tokens": _summary(tokens_per) if tokens_per else None,
        "over_max_len": frac(sum(t > max_len for t in tokens_per), len(tokens_per)) if tokens_per else None,
        "empty_windows": frac(sum(n == 0 for n in nodes_per), len(nodes_per)),
        "semantic_edges": (sem_total, sem_kept, frac(sem_kept, sem_total)),
        "patient_edges": (pat_total, pat_kept, frac(pat_kept, pat_total)),
    }


def print_window_stats(all_stats: list[dict]) -> None:
    """Tabela comparando vários valores de window_lines."""
    head = (f"{'k':>4} {'janelas':>8} {'nós p50':>8} {'nós p95':>8} {'nós máx':>8} "
            f"{'tok p95':>8} {'>max_len':>9} {'vazias':>7} {'arest.sem.':>11} {'arest.pac.':>11}")
    print(head)
    print("-" * len(head))
    for s in all_stats:
        k = s["window_lines"] or "todo"
        tok = s["tokens"]["p95"] if s["tokens"] else "-"
        over = f"{s['over_max_len']:.1%}" if s["over_max_len"] is not None else "-"
        print(f"{k!s:>4} {s['n_windows']:>8} {s['nodes']['p50']:>8} {s['nodes']['p95']:>8} "
              f"{s['nodes']['max']:>8} {tok!s:>8} {over:>9} {s['empty_windows']:>7.1%} "
              f"{s['semantic_edges'][2]:>11.1%} {s['patient_edges'][2]:>11.1%}")


if __name__ == "__main__":
    import argparse

    from diplo_graph03 import config
    from diplo_graph03.preprocessing import dataset

    ap = argparse.ArgumentParser(description="Estatísticas das janelas (use no treino para escolher N e L)")
    ap.add_argument("--data", default=str(config.MACCROBAT_GRAFO_DIR), help="pasta dos casos convertidos")
    ap.add_argument("--lines", type=int, nargs="+", default=[1, 2, 3, 0], help="valores de window_lines (0 = caso inteiro)")
    ap.add_argument("--split", default="train", help="divisão a usar: train, val, test ou all")
    ap.add_argument("--tokenizer", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--no-tokens", action="store_true", help="não conta tokens (não carrega o tokenizer)")
    args = ap.parse_args()

    docs = dataset.load_docs(args.data)
    if args.split != "all":
        ids = set(dataset.get_splits([d.id for d in docs], config.SPLITS_JSON, seed=config.RANDOM_SEED)[args.split])
        docs = [d for d in docs if d.id in ids]
    print(f"{len(docs)} casos ({args.split})")

    counter = None
    if not args.no_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        counter = lambda t: len(tok(t)["input_ids"])

    print_window_stats([window_stats(docs, k, counter) for k in args.lines])