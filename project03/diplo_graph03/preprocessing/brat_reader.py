"""
Leitura do MACCROBAT (formato brat): cada caso tem um .txt (texto) e um .ann (anotações).

No .ann:
    T1<TAB>Age 0 5<TAB>texto              -> entidade (tipo, início e fim em CARACTERES)
    R1<TAB>BEFORE Arg1:T1 Arg2:T2         -> relação entre duas entidades
Entidades descontínuas aparecem como "Sign_symptom 0 5;10 14".
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entity:
    id: str
    label: str
    spans: list          # [(inicio, fim), ...] em caracteres; mais de um se for descontínua
    text: str


@dataclass
class Relation:
    label: str
    arg1: str            # id da entidade de origem
    arg2: str            # id da entidade de destino


@dataclass
class Doc:
    id: str
    text: str
    entities: dict = field(default_factory=dict)    # id -> Entity
    relations: list = field(default_factory=list)   # lista de Relation


def parse_ann(ann_path: Path):
    """Lê um .ann e devolve (entidades por id, lista de relações)."""
    entities, relations = {}, []
    for line in Path(ann_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        key = parts[0]
        if key.startswith("T") and len(parts) >= 2:
            label, span_str = parts[1].split(" ", 1)
            spans = [tuple(map(int, s.split())) for s in span_str.split(";")]
            text = parts[2] if len(parts) > 2 else ""
            entities[key] = Entity(key, label, spans, text)
        elif key.startswith("R") and len(parts) >= 2:
            label, a1, a2 = parts[1].split(" ")[:3]
            relations.append(Relation(label, a1.split(":")[1], a2.split(":")[1]))
    return entities, relations


def load_docs(data_dir) -> list[Doc]:
    """Lê todos os pares .txt/.ann de uma pasta (inclusive subpastas)."""
    data_dir = Path(data_dir)
    docs = []
    for txt in sorted(data_dir.rglob("*.txt")):
        ann = txt.with_suffix(".ann")
        if not ann.exists():
            continue
        # newline="" preserva \r\n: qualquer troca desalinha as posições do .ann
        with txt.open(encoding="utf-8", newline="") as f:
            text = f.read()
        entities, relations = parse_ann(ann)
        docs.append(Doc(txt.stem, text, entities, relations))
    if not docs:
        raise FileNotFoundError(f"nenhum par .txt/.ann encontrado em {data_dir}")
    return docs


def check_offsets(docs) -> int:
    """Conta entidades (contínuas) cujo texto não bate com as posições do .ann."""
    bad = 0
    for d in docs:
        for e in d.entities.values():
            if len(e.spans) == 1:
                s, t = e.spans[0]
                if d.text[s:t] != e.text:
                    bad += 1
    return bad
