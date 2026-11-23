"""
Dados do seq2matrix: cada janela vira um exemplo com a entrada (tokens) e os alvos (nós e arestas).

  1. Tokenização        encode_text, char_to_token_span
  2. Exemplo por janela build_example (entrada + alvos)
  3. Dataset e lote     WindowDataset, collate, make_loader, select_docs

Convenções (valem para o modelo, o pareamento e as perdas):
  * Spans em TOKENS (subtokens do tokenizer do encoder), com início e fim INCLUÍDOS: o texto do nó
    é input_ids[início : fim + 1]. Os tokens especiais ([CLS], [SEP]) nunca fazem parte de um span.
  * Ordem dos slots do gabarito: com patient_slot, o slot 0 é o paciente e os nós reais ocupam os
    slots 1..M, na ordem do texto; sem patient_slot, os nós ocupam 0..M-1. Os slots restantes até
    num_slots são vazios (EMPTY). O modelo prevê as arestas na ordem DELE; o pareamento é que
    alinha as duas ordens (o slot do paciente já está alinhado por construção).
  * edge_target[i, j] = classe da aresta do slot i para o slot j (NO_EDGE_ID se não há aresta).
  * Nós de uma janela com várias menções usam a 1ª menção dentro da janela.

Nada é descartado em silêncio: o que não cabe (span longo, fora do max_len, nós além do max_nodes,
arestas que perdem um nó) é contado em WindowDataset.report. Essa contagem é o TETO de desempenho
que a configuração permite, e entra no relatório.
"""

from __future__ import annotations

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from diplo_graph03 import schema
from diplo_graph03.modeling.seq2matrix.config import Seq2MatrixConfig
from diplo_graph03.preprocessing import windows as W

PAD_ID = 0          # id do [PAD] no PubMedBERT; as posições de padding são mascaradas de qualquer forma
IGNORE = -100       # valor de preenchimento dos alvos de nó (ignorado pelas perdas)

# ================================================================ 1. Tokenização


def build_tokenizer(name: str):
    """Tokenizer rápido do encoder (precisa dos offsets em caracteres)."""
    tok = AutoTokenizer.from_pretrained(name)
    if not tok.is_fast:
        raise ValueError("é preciso um tokenizer 'fast' para obter os offsets")
    return tok


def encode_text(text: str, tokenizer, cfg: Seq2MatrixConfig) -> dict:
    """
    Tokeniza o texto de uma janela. Devolve listas: input_ids, attention_mask, offsets
    (início, fim em caracteres de `text`; (0, 0) nos tokens especiais) e se houve truncamento.
    """
    if cfg.mask_digits:
        text = re.sub(r"\d", "0", text)                     # mantém o tamanho, então os offsets valem
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=cfg.max_len)
    full_len = len(tokenizer(text, verbose=False)["input_ids"])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offsets": [tuple(o) for o in enc["offset_mapping"]],
        "truncated": full_len > cfg.max_len,
    }


def char_to_token_span(offsets, start: int, end: int):
    """
    Span em caracteres -> (token inicial, token final), ambos incluídos. None se nenhum token
    cobre o span (por exemplo, ele ficou depois do max_len).
    """
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    return (idx[0], idx[-1]) if idx else None


# ================================================================ 2. Exemplo por janela


def build_example(window: W.Window, tokenizer, cfg: Seq2MatrixConfig, report: Counter) -> dict:
    """Entrada tokenizada e alvos (tipos, spans e matriz de arestas) de uma janela com grafo ouro."""
    enc = encode_text(window.text, tokenizer, cfg)
    report["windows"] += 1
    report["windows_truncated"] += int(enc["truncated"])

    g = window.local_graph()
    pidx = g.patient_index()
    slots, off = cfg.num_slots, cfg.patient_offset

    # ---- nós: só entram os que têm span válido e cabem nos limites
    kept = {}                                    # índice no grafo da janela -> posição entre os nós mantidos
    types, starts, ends = [], [], []
    for i, node in enumerate(g.nodes):
        if i == pidx:
            continue
        report["nodes_total"] += 1
        tid = schema.NODE_TO_ID.get(node.type)
        if tid is None or tid == schema.EMPTY_ID or not node.spans:
            report["nodes_dropped_invalid"] += 1
            continue
        s, e = min(node.spans)                   # 1ª menção dentro da janela
        ts = char_to_token_span(enc["offsets"], s, e)
        if ts is None:
            report["nodes_dropped_no_token"] += 1          # span depois do max_len, por exemplo
            continue
        if ts[1] - ts[0] + 1 > cfg.max_span_len:
            report["nodes_dropped_span_too_long"] += 1
            continue
        if len(types) >= cfg.max_nodes:
            report["nodes_dropped_over_max_nodes"] += 1
            continue
        kept[i] = len(types)
        types.append(tid)
        starts.append(ts[0])
        ends.append(ts[1])
    report["nodes_kept"] += len(types)

    # ---- arestas
    def slot_of(idx):
        if idx == pidx:
            return 0 if cfg.patient_slot else None
        return kept[idx] + off if idx in kept else None

    edge_target = torch.full((slots, slots), schema.NO_EDGE_ID, dtype=torch.long)
    assigned = {}
    for e in g.edges:
        report["edges_total"] += 1
        cls = schema.EDGE_TO_ID.get(e.type)
        a, b = slot_of(e.src), slot_of(e.dst)
        if cls is None:
            report["edges_dropped_invalid"] += 1
        elif e.src == pidx and not cfg.patient_slot:
            report["edges_patient_skipped"] += 1              # sem slot do paciente, fora do alvo
        elif a is None or b is None:
            report["edges_lost_dropped_node"] += 1
        elif (a, b) in assigned and assigned[(a, b)] != cls:
            report["edges_conflict_same_pair"] += 1           # duas classes no mesmo par: fica a 1ª
        elif (a, b) not in assigned:
            assigned[(a, b)] = cls
            edge_target[a, b] = cls
            report["edges_kept"] += 1

    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "offsets": torch.tensor(enc["offsets"], dtype=torch.long),
        "node_types": torch.tensor(types, dtype=torch.long),
        "node_starts": torch.tensor(starts, dtype=torch.long),
        "node_ends": torch.tensor(ends, dtype=torch.long),
        "edge_target": edge_target,
        "doc_id": window.doc_id,
        "window_index": window.index,
        "window_start": window.start,
        "text": window.text,
    }


# ================================================================ 3. Dataset e lote


def select_docs(docs, ids) -> list:
    """Os casos (Doc) cujo id está em `ids`, ex.: select_docs(docs, splits["train"])."""
    ids = set(ids)
    return [d for d in docs if d.id in ids]


class WindowDataset(Dataset):
    """
    Janelas de uma lista de casos (já no esquema do grafo), prontas para o modelo.
    O filtro por divisão (treino/validação/teste) é feito ANTES, com select_docs: assim todas as
    janelas de um caso ficam no mesmo conjunto.
    """

    def __init__(self, docs, cfg: Seq2MatrixConfig, tokenizer):
        self.cfg = cfg
        self.report = Counter()
        self.examples = []
        for doc in docs:
            for w in W.make_windows(doc, cfg.window_lines):
                self.examples.append(build_example(w, tokenizer, cfg, self.report))
        self.report["cases"] = len(docs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]

    def print_report(self) -> None:
        """Quanto do gabarito cabe na configuração (o teto de desempenho)."""
        r = self.report
        frac = lambda a, b: f"{a / b:.1%}" if b else "-"
        print(f"casos: {r['cases']} | janelas: {r['windows']} | janelas truncadas pelo max_len: {r['windows_truncated']}")
        print(f"nós:      {r['nodes_kept']}/{r['nodes_total']} mantidos ({frac(r['nodes_kept'], r['nodes_total'])})")
        for k in ("nodes_dropped_invalid", "nodes_dropped_no_token",
                  "nodes_dropped_span_too_long", "nodes_dropped_over_max_nodes"):
            if r[k]:
                print(f"   - {k}: {r[k]}")
        print(f"arestas:  {r['edges_kept']}/{r['edges_total']} mantidas ({frac(r['edges_kept'], r['edges_total'])})")
        for k in ("edges_lost_dropped_node", "edges_patient_skipped",
                  "edges_conflict_same_pair", "edges_dropped_invalid"):
            if r[k]:
                print(f"   - {k}: {r[k]}")
        print("(as arestas que cruzam janelas já ficaram de fora do gabarito: ver windows.window_stats)")


def collate(batch: list[dict]) -> dict:
    """
    Junta exemplos num lote, com padding.
      input_ids, attention_mask, token_mask  [B, L]   (token_mask: True onde um span pode apontar)
      offsets                                [B, L, 2]
      num_nodes                              [B]
      node_types, node_starts, node_ends     [B, M]   (preenchidos com IGNORE)
      edge_target                            [B, S, S]
    """
    B = len(batch)
    L = max(x["input_ids"].numel() for x in batch)
    M = max(1, max(x["node_types"].numel() for x in batch))

    input_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.long)
    offsets = torch.zeros(B, L, 2, dtype=torch.long)
    node_types = torch.full((B, M), IGNORE, dtype=torch.long)
    node_starts = torch.full((B, M), IGNORE, dtype=torch.long)
    node_ends = torch.full((B, M), IGNORE, dtype=torch.long)
    num_nodes = torch.zeros(B, dtype=torch.long)

    for b, x in enumerate(batch):
        n, m = x["input_ids"].numel(), x["node_types"].numel()
        input_ids[b, :n] = x["input_ids"]
        attention_mask[b, :n] = x["attention_mask"]
        offsets[b, :n] = x["offsets"]
        node_types[b, :m] = x["node_types"]
        node_starts[b, :m] = x["node_starts"]
        node_ends[b, :m] = x["node_ends"]
        num_nodes[b] = m

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_mask": offsets[..., 1] > offsets[..., 0],        # exclui [CLS], [SEP] e padding
        "offsets": offsets,
        "num_nodes": num_nodes,
        "node_types": node_types,
        "node_starts": node_starts,
        "node_ends": node_ends,
        "edge_target": torch.stack([x["edge_target"] for x in batch]),
        "doc_ids": [x["doc_id"] for x in batch],
        "window_index": [x["window_index"] for x in batch],
        "window_start": [x["window_start"] for x in batch],
        "texts": [x["text"] for x in batch],
    }


def make_loader(dataset: WindowDataset, cfg: Seq2MatrixConfig, shuffle: bool = False,
                seed: int | None = None) -> DataLoader:
    """DataLoader com o tamanho de lote da configuração. A ordem do embaralhamento depende da semente."""
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(cfg.seed if seed is None else seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=gen)
