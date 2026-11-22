"""
Esquema do grafo clínico: fonte única da verdade para as 3 abordagens (seq2seq, seq2tags, seq2matrix).

  1. Tipos de nó e de aresta, e seus ids (a ordem é um CONTRATO: é o índice nos tensores)
  2. Assinatura das arestas (qual tipo de aresta liga quais tipos de nó) e a máscara de esquema
  3. Graph: o objeto que TODA abordagem entrega no fim, para a avaliação ser a mesma

Só usa a biblioteca padrão. As posições (spans) são sempre em CARACTERES do texto do caso,
com início incluído e fim excluído, como no .ann.

Atenção: não reordene NODE_TYPES nem EDGE_TYPES depois de treinar um modelo; os pesos aprendidos
dependem dessa ordem. Classes "vazias" (EMPTY e NO_EDGE) ficam sempre por último.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ================================================================ 1. Nós

# Tipos que o modelo PREVÊ a partir do texto (K = 7).
NODE_TYPES = ("DISEASE", "SYMPTOM", "DRUG", "EXAM", "ANATOMY", "EFFECT", "RESULT")

# O paciente é um nó do grafo, mas não é previsto como span: no seq2matrix é o slot 0.
PATIENT = "PATIENT"
GRAPH_NODE_TYPES = NODE_TYPES + (PATIENT,)

# Classes de saída da cabeça de nós: os 7 tipos + "este slot não tem nó".
EMPTY = "EMPTY"
NODE_CLASSES = NODE_TYPES + (EMPTY,)
N_NODE_CLASSES = len(NODE_CLASSES)                       # 8
NODE_TO_ID = {c: i for i, c in enumerate(NODE_CLASSES)}
EMPTY_ID = NODE_TO_ID[EMPTY]                             # 7 (sempre o último)

# ================================================================ 2. Arestas

# tipo de aresta -> (tipo de nó de origem, tipo de nó de destino). Cada aresta tem UMA assinatura.
EDGE_SIGNATURE = {
    "FAMILY_HISTORY":   (PATIENT, "DISEASE"),
    "PERSONAL_HISTORY": (PATIENT, "DISEASE"),
    "DIAGNOSED_WITH":   (PATIENT, "DISEASE"),
    "HAS_SYMPTOM":      (PATIENT, "SYMPTOM"),
    "TAKES":            (PATIENT, "DRUG"),
    "UNDERGOES":        (PATIENT, "EXAM"),
    "OCCURS_IN":        ("SYMPTOM", "ANATOMY"),
    "AFFECTS":          ("DISEASE", "ANATOMY"),
    "REVEALS":          ("EXAM", "RESULT"),
    "HAS_EFFECT":       ("DRUG", "EFFECT"),
}
EDGE_TYPES = tuple(EDGE_SIGNATURE)                       # 10 tipos, na ordem acima

# Classes de saída da cabeça de arestas: os 10 tipos + "não há aresta entre este par".
NO_EDGE = "NO_EDGE"
EDGE_CLASSES = EDGE_TYPES + (NO_EDGE,)
N_EDGE_CLASSES = len(EDGE_CLASSES)                       # 11
EDGE_TO_ID = {c: i for i, c in enumerate(EDGE_CLASSES)}
NO_EDGE_ID = EDGE_TO_ID[NO_EDGE]                         # 10 (sempre o último)

# As arestas do paciente são reportadas à parte na avaliação (dependem muito do tipo do nó).
PATIENT_EDGE_TYPES = tuple(e for e, (src, _) in EDGE_SIGNATURE.items() if src == PATIENT)
SEMANTIC_EDGE_TYPES = tuple(e for e in EDGE_TYPES if e not in PATIENT_EDGE_TYPES)

# ---------------------------------------------------------------- máscara de esquema

# Tipos que um slot pode ter na hora de aplicar a máscara (PATIENT e EMPTY incluídos).
MASK_TYPES = GRAPH_NODE_TYPES + (EMPTY,)
MASK_TYPE_TO_ID = {t: i for i, t in enumerate(MASK_TYPES)}


def allowed_edge_types(src_type: str, dst_type: str) -> tuple[str, ...]:
    """Tipos de aresta permitidos de src_type para dst_type (sem contar NO_EDGE)."""
    return tuple(e for e, sig in EDGE_SIGNATURE.items() if sig == (src_type, dst_type))


def edge_mask(src_type: str, dst_type: str) -> list[bool]:
    """Lista de bool sobre EDGE_CLASSES: True = permitido. NO_EDGE sempre é permitido."""
    allowed = set(allowed_edge_types(src_type, dst_type))
    return [(c in allowed) or c == NO_EDGE for c in EDGE_CLASSES]


def edge_mask_table() -> list[list[list[bool]]]:
    """
    Tabela [tipo_origem][tipo_destino][classe_de_aresta] (índices em MASK_TYPES e EDGE_CLASSES).
    O modelo converte para tensor uma vez e usa para mascarar os logits de arestas.
    """
    return [[edge_mask(s, d) for d in MASK_TYPES] for s in MASK_TYPES]


# ================================================================ 3. Graph


@dataclass
class GraphNode:
    type: str                                   # um de GRAPH_NODE_TYPES
    spans: list = field(default_factory=list)   # [(início, fim), ...] em caracteres do caso
    text: str = ""


@dataclass
class GraphEdge:
    type: str                                   # um de EDGE_TYPES
    src: int                                    # índice em Graph.nodes
    dst: int                                    # índice em Graph.nodes


@dataclass
class Graph:
    """Grafo de um caso (ou de uma janela). É o formato de saída comum às 3 abordagens."""

    nodes: list = field(default_factory=list)   # lista de GraphNode
    edges: list = field(default_factory=list)   # lista de GraphEdge
    doc_id: str = ""

    # ------------------------------------------------------------ consulta

    def patient_index(self) -> int | None:
        """Índice do nó PATIENT (None se não houver)."""
        for i, n in enumerate(self.nodes):
            if n.type == PATIENT:
                return i
        return None

    def edge_validity(self) -> tuple[int, int]:
        """(arestas que respeitam o esquema, total de arestas): base da 'taxa de validade de esquema'."""
        ok = 0
        for e in self.edges:
            if not (0 <= e.src < len(self.nodes) and 0 <= e.dst < len(self.nodes)):
                continue
            if EDGE_SIGNATURE.get(e.type) == (self.nodes[e.src].type, self.nodes[e.dst].type):
                ok += 1
        return ok, len(self.edges)

    def problems(self) -> list[str]:
        """Lista de problemas estruturais (vazia se o grafo está coerente com o esquema)."""
        out = []
        n = len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node.type not in GRAPH_NODE_TYPES:
                out.append(f"nó {i}: tipo desconhecido {node.type!r}")
            if node.type != PATIENT and not node.spans:
                out.append(f"nó {i} ({node.type}): sem posição no texto")
            for s, e in node.spans:
                if not (0 <= s < e):
                    out.append(f"nó {i}: span inválido ({s}, {e})")
        if sum(1 for x in self.nodes if x.type == PATIENT) > 1:
            out.append("mais de um nó PATIENT")
        for j, e in enumerate(self.edges):
            if e.type not in EDGE_SIGNATURE:
                out.append(f"aresta {j}: tipo desconhecido {e.type!r}")
            elif not (0 <= e.src < n and 0 <= e.dst < n):
                out.append(f"aresta {j}: índice fora do intervalo ({e.src}, {e.dst})")
            elif EDGE_SIGNATURE[e.type] != (self.nodes[e.src].type, self.nodes[e.dst].type):
                out.append(
                    f"aresta {j}: {e.type} não liga {self.nodes[e.src].type} -> {self.nodes[e.dst].type}")
        return out

    # ------------------------------------------------------------ conversões

    def to_dict(self) -> dict:
        """Dicionário serializável em JSON (útil para o seq2seq gerar e parsear o grafo)."""
        return {
            "doc_id": self.doc_id,
            "nodes": [{"type": n.type, "spans": [list(s) for s in n.spans], "text": n.text}
                      for n in self.nodes],
            "edges": [{"type": e.type, "src": e.src, "dst": e.dst} for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        nodes = [GraphNode(n["type"], [tuple(s) for s in n.get("spans", [])], n.get("text", ""))
                 for n in d.get("nodes", [])]
        edges = [GraphEdge(e["type"], int(e["src"]), int(e["dst"])) for e in d.get("edges", [])]
        return cls(nodes, edges, d.get("doc_id", ""))

    @classmethod
    def from_brat(cls, entities: dict, relations: list, doc_id: str = "") -> "Graph":
        """
        Monta o grafo ouro a partir de entidades e relações já convertidas para o esquema
        (saída do dataset.convert_folder). Aceita qualquer objeto com os atributos
        `id, label, spans, text` (entidade) e `label, arg1, arg2` (relação).
        Os nós saem ordenados pela posição da 1ª menção (ordem do texto).
        """
        ents = [e for e in entities.values() if e.label in GRAPH_NODE_TYPES]
        ents.sort(key=lambda e: min((s for s, _ in e.spans), default=-1))
        index = {e.id: i for i, e in enumerate(ents)}
        nodes = [GraphNode(e.label, [tuple(s) for s in e.spans], e.text) for e in ents]
        edges = [GraphEdge(r.label, index[r.arg1], index[r.arg2])
                 for r in relations
                 if r.label in EDGE_SIGNATURE and r.arg1 in index and r.arg2 in index]
        return cls(nodes, edges, doc_id)