"""
edge_clasifier.py -- Classificador de arestas (abordagem seq2tags) usando embeddings PubMedBERT.

Espelha o node_clasifier.py: mesma estrutura (StandardScaler + LogisticRegression), só que
treinando sobre pares de entidades em vez de subtokens.

De onde vem X/y: features.build_edge_dataset(docs, token_data, ...), que já:
  - gera os pares (positivos = toda relação do .ann; negativos = pares próximos sem relação);
  - monta X = [v1 ; v2 ; |v1-v2| ; distância] a partir dos embeddings médios de cada entidade;
  - rotula com o tipo da relação (schema.EDGE_TYPES) ou schema.NO_EDGE.

IMPORTANTE (combinar com o time antes de rodar ponta-a-ponta): este dataset usa as entidades
OURO (`Doc.entities`), não as entidades previstas pelo NodeClassifier. Ainda falta decidir o
formato comum de entidade entre os 2 estágios (node_clasifier.bio_to_entities não devolve
`"text"`, por exemplo) antes de encadear nós previstos -> arestas previstas de verdade. Por
ora, treinamos e avaliamos com nós ouro, igual fizemos com o classificador de nós isoladamente.

O `step3_attributes_and_graph.py` (quantidades/unidades nas arestas) fica para depois, por
decisão do time -- quando houver uma classe própria de atributo que não é obrigatória em toda
aresta. Por enquanto, `build_graph_for_doc` monta `schema.Graph`/`GraphEdge` só com (type, src, dst).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

try:
    from diplo_graph03 import schema
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from diplo_graph03 import schema


# ------------------------------------------------------------------ classificador
class EdgeClassifier:
    """
    Wrapper fino sobre LogisticRegression, mesmo espírito do NodeClassifier: StandardScaler
    (embeddings densos e com escalas variadas) + solver 'lbfgs' (mais rápido que 'saga' aqui).

    Parâmetros
    ----------
    C         : regularização L2 (maior = menos regularização)
    max_iter  : iterações do solver
    verbose   : 1 para imprimir o progresso de convergência
    """

    def __init__(self, C: float = 1.0, max_iter: int = 200, verbose: int = 0):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",   # NO_EDGE domina o dataset, igual "O" dominava nos nós
            solver="lbfgs",
            verbose=verbose,
        )

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

    @property
    def classes_(self):
        return self.clf.classes_

    def predict_masked(self, X, type_pairs):
        """
        Como predict(), mas para cada par (i) só escolhe entre os tipos de aresta
        estruturalmente válidos para `type_pairs[i] = (tipo_src, tipo_dst)`
        (`schema.allowed_edge_types(tipo_src, tipo_dst)`) + schema.NO_EDGE -- igual ao
        `predict_edges_with_proba` do P2, que restringia via VALID_SCHEMA_PAIRS.

        Sem essa máscara, o argmax puro às vezes escolhe um tipo que não bate com os
        tipos dos nós (ex.: UNDERGOES entre PATIENT -> DRUG), o que depois aparece como
        "problema estrutural" na hora de montar o Graph (schema.Graph.problems()).

        type_pairs: lista/array de (tipo_src, tipo_dst), mesmo tamanho e ordem de X
                    (ver `type_pairs_from_meta`, que monta isso a partir de `meta`).
        """
        proba = self.predict_proba(X)
        classes = list(self.classes_)
        class_idx = {c: i for i, c in enumerate(classes)}

        out = []
        for row_proba, (t1, t2) in zip(proba, type_pairs):
            allowed = set(schema.allowed_edge_types(t1, t2)) | {schema.NO_EDGE}
            allowed_idx = [class_idx[c] for c in allowed if c in class_idx]
            if not allowed_idx:
                # nenhuma classe conhecida é válida para esse par de tipos -> NO_EDGE
                out.append(schema.NO_EDGE)
                continue
            best = max(allowed_idx, key=lambda i: row_proba[i])
            out.append(classes[best])
        return np.array(out, dtype=object)

    def evaluate(self, y_true, y_pred) -> dict:
        """
        P/R/F1 por tipo de aresta + "micro" calculado SÓ sobre as arestas positivas
        (ignora schema.NO_EDGE nas médias) -- o mesmo critério usado no P2
        (pipeline_completo.ipynb, edge_metrics_clean).
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        classes = sorted(set(y_true) | set(y_pred))
        pos_classes = [c for c in classes if c != schema.NO_EDGE]

        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0)
        report = {c: {"p": float(pp), "r": float(rr), "f1": float(ff), "support": int(ss)}
                  for c, pp, rr, ff, ss in zip(classes, p, r, f1, support)}

        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=pos_classes, average="micro", zero_division=0)
        n_pos = sum(report[c]["support"] for c in pos_classes)
        report["micro (só arestas positivas)"] = {
            "p": float(p_mic), "r": float(r_mic), "f1": float(f1_mic), "support": int(n_pos)}
        return report

    def print_report(self, report: dict) -> None:
        print(f"{'tipo':30s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>6s}")
        for typ, m in report.items():
            print(f"{typ:30s} {m['p']:6.3f} {m['r']:6.3f} {m['f1']:6.3f} {m['support']:6d}")


# ------------------------------------------------------------------ máscara de esquema
def type_pairs_from_meta(docs, meta):
    """
    Para cada linha de `meta` (doc_id, id_a, id_b), devolve (tipo_a, tipo_b) -- os tipos
    (schema.GRAPH_NODE_TYPES) das entidades OURO correspondentes, na mesma ordem de `meta`.
    Entrada de `EdgeClassifier.predict_masked`.

    OBS: usa `doc.entities[id].label`, que já vem convertido para o vocabulário do schema
    novo (NODE_MAP em dataset.py) -- é o mesmo label usado em `entities_to_graph_nodes`.
    """
    by_id = {str(d.id): d for d in docs}
    pairs = []
    for did, a, b in meta:
        d = by_id[str(did)]
        pairs.append((d.entities[a].label, d.entities[b].label))
    return pairs


# ------------------------------------------------------------------ montagem do Graph (ouro)
def entities_to_graph_nodes(doc):
    """
    Nós (ordenados pela 1ª menção) + índice id_entidade -> posição no Graph, a partir das
    entidades OURO do doc (mesma ordenação que schema.Graph.from_brat usa).
    """
    ents = [e for e in doc.entities.values() if e.label in schema.GRAPH_NODE_TYPES]
    ents.sort(key=lambda e: min((s for s, _ in e.spans), default=-1))
    index = {e.id: i for i, e in enumerate(ents)}
    nodes = [schema.GraphNode(e.label, list(e.spans), e.text) for e in ents]
    return nodes, index


def build_graph_for_doc(doc, meta, y_pred) -> "schema.Graph":
    """
    Monta um schema.Graph PREVISTO para 1 doc: nós ouro (ver nota no topo do módulo sobre
    ainda não termos nós previstos encadeados) + arestas previstas (ignora schema.NO_EDGE).

    meta, y_pred: saída de features.build_edge_dataset (meta) e EdgeClassifier.predict (y_pred),
    já filtrados/alinhados (mesma ordem, cobrindo pelo menos este doc -- pode conter outros docs
    também, a função filtra pelo doc.id sozinha).
    """
    nodes, index = entities_to_graph_nodes(doc)
    edges = []
    for (did, a, b), label in zip(meta, y_pred):
        if str(did) != str(doc.id) or label == schema.NO_EDGE:
            continue
        if a in index and b in index:
            edges.append(schema.GraphEdge(label, index[a], index[b]))
    return schema.Graph(nodes, edges, doc.id)
