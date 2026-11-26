"""
node_clasifier.py -- Classificador de nós (abordagem seq2tags) usando embeddings PubMedBERT.

Parte da reorganização em diplo_graph03/modeling/seq2tags/, ao lado de features.py
(BertEmbedder + build_token_dataset/build_edge_dataset).

O que mudou em relação à versão anterior (project03/models/classificar_nos/):
  - Os labels de entidade já chegam mapeados para o schema do grafo (PATIENT, DISEASE,
    SYMPTOM, ANATOMY, DRUG, EXAM, RESULT, EFFECT) -- esse mapeamento agora é feito em
    preprocessing/dataset.py (NODE_MAP + convert_folder), não mais aqui. Isso também
    resolve boa parte do desbalanceamento extremo entre tipos que víamos com os ~40
    rótulos originais do MACCROBAT (agora são só os 7 tipos de schema.NODE_TYPES).
  - evaluation.py (compartilhado entre seq2tags/seq2seq/seq2matrix) continua com a
    mesma API de antes (entity_f1/print_report sobre arrays BIO + doc ids), então os
    métodos evaluate()/print_report() da classe foram mantidos.
  - schema.py define o Graph (GraphNode/GraphEdge) usado pelas 3 abordagens no fim do
    pipeline; bio_to_entities ainda devolve dicts simples (não GraphNode) -- a conversão
    para Graph.from_brat-like fica para quando o classificador de arestas estiver pronto.

Uso típico (ver node_pipeline.ipynb):

    from diplo_graph03.modeling.seq2tags.node_clasifier import (
        NodeClassifier, fix_bio, bio_to_entities,
    )

    clf = NodeClassifier().fit(train["X"], train["y"])
    pred = clf.predict(val["X"])
    rep = clf.evaluate(val["y"], pred, val["doc"])
    clf.print_report(rep)
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# import de pacote (diplo_graph03 na raiz do projeto); fallback para quem roda este
# arquivo fora do pacote (ex.: notebook sem instalar o projeto)
try:
    from diplo_graph03 import evaluation, schema
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from diplo_graph03 import evaluation, schema


# ------------------------------------------------------------------ classificador
class NodeClassifier:
    """
    Wrapper fino sobre LogisticRegression que recebe diretamente o X denso
    (numpy array [n_subtokens, 768], saída de features.BertEmbedder) e as tags
    BIO como y (já no schema do grafo, graças a preprocessing/dataset.py).

    Por que lbfgs + StandardScaler (e não saga "cru"): embeddings de BERT são densos
    e com escalas variadas; 'saga' foi pensado para dados esparsos de alta dimensão
    (TF-IDF/DictVectorizer, como no P2). Sem normalizar, qualquer solver gradiente
    demora muito mais para convergir (ou nem converge dentro do max_iter).

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
            class_weight="balanced",   # compensa o excesso de tokens "O"
            solver="lbfgs",            # rápido para dados densos (diferente de saga)
            verbose=verbose,
        )

    def fit(self, X, y):
        """Normaliza e treina sobre todos os subtokens de treino."""
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def predict(self, X):
        """Retorna as tags BIO preditas (aplica o mesmo scaler do treino)."""
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        """Probabilidades por classe (útil para inspeção)."""
        return self.clf.predict_proba(self.scaler.transform(X))

    @property
    def classes_(self):
        return self.clf.classes_

    def evaluate(self, y_true, y_pred, doc) -> dict:
        """
        Avaliação por ENTIDADE (não por subtoken) usando evaluation.entity_f1.
        Retorna {tipo: {p, r, f1, support}, "micro": {...}}.
        """
        return evaluation.entity_f1(y_true, y_pred, doc)

    def print_report(self, report: dict) -> None:
        evaluation.print_report(report)


# ------------------------------------------------------------------ pós-processamento BIO
def fix_bio(tags: list[str]) -> list[str]:
    """
    Corrige BIO inválido: um I-X que não continua um B-X/I-X do mesmo tipo
    vira B-X. Agnóstico ao dataset -- não precisou mudar na reorganização.
    """
    out = list(tags)
    for i, tag in enumerate(out):
        if tag.startswith("I-"):
            prev = out[i - 1] if i > 0 else "O"
            if prev == "O" or prev[2:] != tag[2:]:
                out[i] = "B-" + tag[2:]
    return out


def bio_to_entities(offsets, tags: list[str]) -> list[dict]:
    """
    Converte tags BIO em entidades com posição de caractere.

    Parâmetros
    ----------
    offsets : array [n_subtokens, 2] com (char_start, char_end) de cada subtoken
              (campo "offsets" do features.load_token_dataset)
    tags    : lista de tags BIO alinhadas aos subtokens (já com os rótulos do
              schema do grafo: DISEASE, SYMPTOM, ANATOMY, DRUG, EXAM, RESULT, PATIENT)

    Retorna
    -------
    lista de dicts: {"label", "char_start", "char_end", "token_start", "token_end"}

    Nota: se schema.py tiver uma classe Node própria (ex.: para alimentar o Graph
    compartilhado), é só trocar o dict de saída daqui por schema.Node(...) -- avise
    quando puder compartilhar schema.py que eu ajusto.
    """
    entities = []
    n = len(tags)
    k = 0
    while k < n:
        tag = tags[k]
        if tag == "O":
            k += 1
            continue
        typ = tag[2:]          # remove "B-" ou "I-"
        start_k = k
        while k + 1 < n and tags[k + 1] == "I-" + typ:
            k += 1
        entities.append({
            "label":       typ,
            "char_start":  int(offsets[start_k][0]),
            "char_end":    int(offsets[k][1]),
            "token_start": start_k,
            "token_end":   k + 1,   # exclusivo
        })
        k += 1
    return entities
