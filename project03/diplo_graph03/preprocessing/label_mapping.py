"""
Converte o MACCROBAT (brat .txt/.ann) para o esquema do nosso grafo.

Uso:
    python -m diplo_graph03.preprocessing.label_mapping PASTA_MACCROBAT PASTA_SAIDA
    python -m diplo_graph03.preprocessing.label_mapping    # sem argumentos: usa os caminhos do config.py

Para cada par .txt/.ann da entrada, grava na saída um par com o mesmo nome:
  - .txt: cópia idêntica (as posições em caracteres continuam valendo);
  - .ann: entidades com os tipos de nó do grafo e relações com os tipos de aresta do grafo
          (só linhas T e R, com ids T, então o brat_reader.py lê a saída sem problemas).

Nós do grafo:    DISEASE, SYMPTOM, DRUG, EXAM, ANATOMY, EFFECT, RESULT, PATIENT
Arestas do grafo:
    PATIENT -> DISEASE : FAMILY_HISTORY | PERSONAL_HISTORY | DIAGNOSED_WITH
    PATIENT -> SYMPTOM : HAS_SYMPTOM
    SYMPTOM -> ANATOMY : OCCURS_IN
    DISEASE -> ANATOMY : AFFECTS
    PATIENT -> DRUG    : TAKES
    PATIENT -> EXAM    : UNDERGOES
    EXAM    -> RESULT  : REVEALS
    DRUG    -> EFFECT  : HAS_EFFECT

O MACCROBAT não tem relações semânticas desse tipo (só MODIFY, BEFORE, AFTER, IDENTICAL...),
então as ARESTAS SÃO DERIVADAS POR REGRAS a partir dos tipos das entidades.

Particularidades do formato do MACCROBAT tratadas aqui:
  * linhas "E1<TAB>Label:T4": eventos que apontam para uma entidade T. As relações usam
    ids E (Arg1:E7), então é preciso traduzir E -> T antes de usá-las;
  * IDENTICAL (correferência): "symptoms" IDENTICAL "palpitations" etc.;
  * atributo POLARITY: NEG bloqueia a aresta do paciente (qualquer tipo); MAYBE_POS/MAYBE_NEG
    ("possibility of a sarcoma") bloqueiam só SYMPTOM/DISEASE. O nó em si continua existindo;
  * trechos de histórico: SYMPTOM dentro deles é descartado ("previously healthy" contém "healthy"),
    e DISEASE dentro deles vira PERSONAL_/FAMILY_HISTORY ("history of pulmonary tuberculosis");
  * CAUSE de DRUG para SYMPTOM/DISEASE: o alvo vira EFFECT e a aresta HAS_EFFECT é criada
    (única fonte de EFFECT no MACCROBAT);
  * EXAM que é SUB_PROCEDURE de outro EXAM ("sinus rhythm" dentro de "electrocardiogram") não ganha
    UNDERGOES do paciente; o resultado dele passa a ser revelado também pelo exame pai;
  * RESULT sem nenhum EXAM ligado é descartado;
  * menções duplicadas do mesmo conceito viram UM nó com várias posições no .ann
    (ex.: "T7 SYMPTOM 78 90;1631 1643"); o brat_reader/build_datasets já lidam com isso;
  * PATIENT: 1ª menção de Age/Sex; se não houver, nó sintético ancorado na 1ª palavra do texto.
"""

import re
import shutil
import sys
from collections import Counter, deque
from pathlib import Path

from .brat_reader import Entity, Relation

# ---------------------------------------------------------------- nós

# rótulo MACCROBAT -> tipo de nó do grafo. O que não estiver aqui é descartado.
NODE_MAP = {
    "Sign_symptom":         "SYMPTOM",
    "Disease_disorder":     "DISEASE",
    "Biological_structure": "ANATOMY",
    "Medication":           "DRUG",
    "Diagnostic_procedure": "EXAM",
    "Lab_value":            "RESULT",
}
# Não existe equivalente no MACCROBAT para EFFECT; ele sai sem nós.
# Therapeutic_procedure também é descartado (o grafo não tem esse tipo).

PATIENT_SOURCES = ("Age", "Sex")

FAMILY_HISTORY_LABEL = "Family_history"
PERSONAL_HISTORY_LABELS = ("History", "Personal_background")
HISTORY_LABELS = {FAMILY_HISTORY_LABEL, *PERSONAL_HISTORY_LABELS}

# Nós destes tipos dentro de um trecho de histórico são descartados ("healthy" dentro de
# "previously healthy"). DISEASE fica de fora de propósito: "history of tuberculosis" contém a doença,
# que deve virar PERSONAL_HISTORY, não sumir.
DROP_INSIDE_HISTORY_TYPES = {"SYMPTOM"}

# Entidades intermediárias (Detailed_description, Severity, Coreference...) podem ficar entre dois nós:
# ligamos dois nós se houver um caminho de MODIFY com no máximo MAX_HOPS intermediárias.
MAX_HOPS = 2

# RESULT só faz sentido ligado a um EXAM (REVEALS). RESULT sem nenhuma aresta REVEALS é descartado
# (ex.: "grade 2/6" modificando um sintoma).
PRUNE_ORPHAN_RESULTS = True

# Fusão de menções duplicadas do mesmo conceito em UM nó (com várias posições no .ann):
#  - sempre: entidades ligadas por IDENTICAL (mesmo tipo de nó);
#  - por texto igual (sem diferenciar maiúsculas), só nestes tipos. SYMPTOM, EXAM e RESULT ficam de
#    fora de propósito: "pain" na cabeça e "pain" na perna, ou o ECG antes e depois da ablação, são
#    ocorrências diferentes. Para esses tipos só vale o IDENTICAL.
MERGE_BY_TEXT_TYPES = {"ANATOMY", "DISEASE", "DRUG"}

# ---------------------------------------------------------------- arestas

# (tipo origem, tipo destino) -> aresta. A direção do MODIFY no .ann é ignorada;
# a aresta sai orientada como no nosso grafo.
MODIFY_RULES = {
    ("SYMPTOM", "ANATOMY"): "OCCURS_IN",
    ("DISEASE", "ANATOMY"): "AFFECTS",
    ("EXAM", "RESULT"):     "REVEALS",
}

# "todo nó do tipo X pertence ao paciente do caso"
PATIENT_RULES = {
    "DISEASE": "DIAGNOSED_WITH",    # vira FAMILY_/PERSONAL_HISTORY se houver Family_history/History
    "SYMPTOM": "HAS_SYMPTOM",
    "DRUG":    "TAKES",
    "EXAM":    "UNDERGOES",
}
# POLARITY NEG: nenhum tipo ganha aresta do paciente ("no recurrence of palpitations",
# "no regular medication"). MAYBE_POS / MAYBE_NEG: só estes tipos ("possibility of a sarcoma").
UNCERTAIN_BLOCKS = {"DISEASE", "SYMPTOM"}

# CAUSE entre estes tipos vira DRUG --HAS_EFFECT--> EFFECT (o alvo é reclassificado como EFFECT).
EFFECT_SOURCE = "DRUG"
EFFECT_TARGETS = {"SYMPTOM", "DISEASE"}


# ---------------------------------------------------------------- leitura do .ann

def parse_ann_full(path):
    """Devolve (entidades {T: Entity}, relações com ids T, polaridade {T: NEG|MAYBE_POS|MAYBE_NEG})."""
    entities, events, raw_rel, attrs = {}, {}, [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("*"):       # "*" = OVERLAP n-ário (temporal)
            continue
        parts = line.split("\t")
        key = parts[0]
        if key.startswith("T") and len(parts) >= 2:
            label, span_str = parts[1].split(" ", 1)
            spans = [tuple(map(int, s.split())) for s in span_str.split(";")]
            entities[key] = Entity(key, label, spans, parts[2] if len(parts) > 2 else "")
        elif key.startswith("E") and len(parts) >= 2:
            events[key] = parts[1].split(" ")[0].split(":")[1]          # E1 -> T4
        elif key.startswith("R") and len(parts) >= 2:
            f = parts[1].split(" ")
            raw_rel.append((f[0], f[1].split(":")[1], f[2].split(":")[1]))
        elif key.startswith("A") and len(parts) >= 2:
            f = parts[1].split(" ")
            if len(f) >= 3:
                attrs.append((f[0], f[1], f[2]))

    resolve = lambda i: events.get(i, i)
    relations = [Relation(lab, resolve(a), resolve(b)) for lab, a, b in raw_rel]
    polarity = {resolve(t): val for name, t, val in attrs
                if name == "POLARITY" and val in ("NEG", "MAYBE_POS", "MAYBE_NEG")}
    return entities, relations, polarity


# ---------------------------------------------------------------- regras

def reachable_nodes(src, adj, ident, nodes, passable, max_hops):
    """
    Nós alcançáveis a partir de `src` por relações `adj` (MODIFY), passando só por entidades
    `passable` (até max_hops). Arestas IDENTICAL (`ident`) não custam salto: a correferência
    ("valve" = "tricuspid valve") é tratada como a mesma entidade.
    """
    found, seen = set(), {src}
    queue = deque([(src, 0)])
    while queue:
        cur, hops = queue.popleft()
        for nxt in ident.get(cur, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt in nodes and cur not in nodes:
                found.add(nxt)                       # a correferência resolveu para um nó
            else:
                queue.append((nxt, hops))            # mesma entidade: continua de lá
        for nxt in adj.get(cur, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt in nodes:
                found.add(nxt)
            elif nxt in passable and hops < max_hops:
                queue.append((nxt, hops + 1))
    return found


def make_patient(entities, text):
    """(id, spans, texto, sintético?) do PATIENT do caso."""
    for lab in PATIENT_SOURCES:
        cand = [e for e in entities.values() if e.label == lab]
        if cand:
            first = min(cand, key=lambda e: min(s for s, _ in e.spans))
            return first.id, first.spans, first.text, False
    m = re.search(r"\S+", text)
    start, end = (m.start(), m.end()) if m else (0, 0)
    nums = [int(i[1:]) for i in entities if i[1:].isdigit()]
    return f"T{max(nums, default=0) + 1}", [(start, end)], text[start:end], True


def merge_mentions(nodes, edges, entities, relations, info):
    """Funde menções duplicadas em um nó só: spans somados, arestas redirecionadas e sem repetição."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # componentes de IDENTICAL sobre TODAS as entidades (inclui Coreference no meio da cadeia)
    comp = {e: e for e in entities}

    def cfind(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    for r in relations:
        if r.label == "IDENTICAL" and r.arg1 in comp and r.arg2 in comp:
            comp[cfind(r.arg1)] = cfind(r.arg2)

    seen_ident, seen_text = {}, {}
    for nid, (ntype, _, text) in nodes.items():
        if ntype == "PATIENT":
            continue
        if nid in entities:
            key = (cfind(nid), ntype)
            if key in seen_ident:
                union(nid, seen_ident[key])
            else:
                seen_ident[key] = nid
        if ntype in MERGE_BY_TEXT_TYPES:
            key = (ntype, re.sub(r"\s+", " ", text.lower().strip()))
            if key in seen_text:
                union(nid, seen_text[key])
            else:
                seen_text[key] = nid

    clusters = {}
    for nid in nodes:
        clusters.setdefault(find(nid), []).append(nid)

    rep, merged = {}, {}
    for members in clusters.values():
        spans = sorted({sp for m in members for sp in nodes[m][1]})
        first = min(members, key=lambda m: min(s for s, _ in nodes[m][1]))
        texts = list(dict.fromkeys(nodes[m][2] for m in sorted(members, key=lambda m: min(s for s, _ in nodes[m][1]))))
        merged[first] = (nodes[first][0], spans, " | ".join(texts))
        for m in members:
            rep[m] = first
        info["nós fundidos (menções duplicadas)"] += len(members) - 1

    new_nodes = {nid: merged[nid] for nid in nodes if nid in merged}
    new_edges = sorted({(lab, rep[x], rep[y]) for lab, x, y in edges if rep[x] != rep[y]})
    return new_nodes, new_edges


def derive(entities, relations, polarity, text, max_hops=MAX_HOPS):
    """Devolve (nós {id: (tipo, spans, texto)}, arestas [(rótulo, id1, id2)], info)."""
    info = Counter()

    hist_spans = [sp for e in entities.values() if e.label in HISTORY_LABELS for sp in e.spans]
    inside_history = lambda spans: all(
        any(h0 <= s and e <= h1 for h0, h1 in hist_spans) for s, e in spans)

    nodes = {}
    for eid, e in entities.items():
        if e.label not in NODE_MAP:
            continue
        if NODE_MAP[e.label] in DROP_INSIDE_HISTORY_TYPES and inside_history(e.spans):
            info["nós descartados (dentro de trecho de histórico)"] += 1
            continue
        nodes[eid] = (NODE_MAP[e.label], e.spans, e.text)

    patient_id, spans, ptext, synthetic = make_patient(entities, text)
    nodes[patient_id] = ("PATIENT", spans, ptext)
    info["casos com PATIENT sintético"] += int(synthetic)

    # entidades-ponte: não são nós nem marcadores de histórico/paciente
    excluded = HISTORY_LABELS | set(PATIENT_SOURCES)
    passable = {eid for eid, e in entities.items()
                if eid not in nodes and e.label not in excluded}

    adj_modify, adj_all, ident = {}, {}, {}
    sub_parent = {}      # EXAM "achado"/sub-procedimento -> EXAM pai (SUB_PROCEDURE: Arg1 = filho)
    for r in relations:
        if r.arg1 not in entities or r.arg2 not in entities:
            info["relações com id não resolvido"] += 1
            continue
        if r.label == "IDENTICAL":
            ident.setdefault(r.arg1, set()).add(r.arg2)
            ident.setdefault(r.arg2, set()).add(r.arg1)
            continue
        if (r.label == "SUB_PROCEDURE" and r.arg1 in nodes and r.arg2 in nodes
                and nodes[r.arg1][0] == nodes[r.arg2][0] == "EXAM"):
            sub_parent[r.arg1] = r.arg2
        for adj, ok in ((adj_all, True), (adj_modify, r.label == "MODIFY")):
            if ok:
                adj.setdefault(r.arg1, set()).add(r.arg2)
                adj.setdefault(r.arg2, set()).add(r.arg1)

    edges = set()

    # 0) efeito de droga: CAUSE(DRUG -> SYMPTOM/DISEASE) reclassifica o alvo como EFFECT
    for r in relations:
        if (r.label == "CAUSE" and r.arg1 in nodes and r.arg2 in nodes
                and nodes[r.arg1][0] == EFFECT_SOURCE and nodes[r.arg2][0] in EFFECT_TARGETS):
            _, sp, tx = nodes[r.arg2]
            nodes[r.arg2] = ("EFFECT", sp, tx)
            edges.add(("HAS_EFFECT", r.arg1, r.arg2))
            info["efeitos de droga (CAUSE)"] += 1

    # 1) regras por par de tipos
    for a in nodes:
        for b in reachable_nodes(a, adj_modify, ident, nodes, passable, max_hops):
            label = MODIFY_RULES.get((nodes[a][0], nodes[b][0]))
            if label:
                edges.add((label, a, b))

    # o resultado de um sub-procedimento ("normal sinus rhythm") também é revelado pelo exame pai
    for lab, a, b in list(edges):
        if lab == "REVEALS" and a in sub_parent:
            edges.add(("REVEALS", sub_parent[a], b))

    # 2) histórico: doença ligada por relação/ponte OU contida no trecho de histórico
    family, personal = set(), set()
    for eid, e in entities.items():
        kind = (family if e.label == FAMILY_HISTORY_LABEL
                else personal if e.label in PERSONAL_HISTORY_LABELS else None)
        if kind is None:
            continue
        for b in reachable_nodes(eid, adj_all, ident, nodes, passable, max_hops):
            if nodes[b][0] == "DISEASE":
                kind.add(b)
        for nid, (ntype, nspans, _) in nodes.items():
            if ntype == "DISEASE" and all(
                    any(a <= s0 and t0 <= b0 for a, b0 in e.spans) for s0, t0 in nspans):
                kind.add(nid)

    # 3) paciente
    for nid, (ntype, _, _) in nodes.items():
        label = PATIENT_RULES.get(ntype)
        if label is None:
            continue
        pol = polarity.get(nid)
        if pol == "NEG" or (pol in ("MAYBE_POS", "MAYBE_NEG") and ntype in UNCERTAIN_BLOCKS):
            info[f"arestas do paciente puladas (polaridade {pol})"] += 1
            continue
        if ntype == "EXAM" and nid in sub_parent:
            info["UNDERGOES pulado (achado/sub-procedimento de outro exame)"] += 1
            continue
        if ntype == "DISEASE":
            label = ("FAMILY_HISTORY" if nid in family
                     else "PERSONAL_HISTORY" if nid in personal else label)
        edges.add((label, patient_id, nid))

    # 4) RESULT órfão
    if PRUNE_ORPHAN_RESULTS:
        with_reveals = {b for lab, _, b in edges if lab == "REVEALS"}
        for nid in [n for n, (t, _, _) in nodes.items() if t == "RESULT" and n not in with_reveals]:
            del nodes[nid]
            info["RESULT órfãos descartados"] += 1

    # 5) fusão de menções duplicadas
    nodes, edges = merge_mentions(nodes, edges, entities, relations, info)
    return nodes, edges, info


# ---------------------------------------------------------------- saída

def write_ann(path, nodes, edges):
    lines = []
    for nid, (ntype, spans, text) in nodes.items():
        span_str = ";".join(f"{s} {e}" for s, e in spans)
        lines.append(f"{nid}\t{ntype} {span_str}\t{text}")
    for i, (label, a, b) in enumerate(edges, 1):
        lines.append(f"R{i}\t{label} Arg1:{a} Arg2:{b}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_folder(in_dir, out_dir) -> Counter:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    stats = Counter()
    for txt in sorted(in_dir.rglob("*.txt")):
        ann = txt.with_suffix(".ann")
        if not ann.exists():
            continue
        dest = out_dir / txt.relative_to(in_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(txt, dest)                       # cópia byte a byte (preserva \r\n)

        entities, relations, polarity = parse_ann_full(ann)
        with txt.open(encoding="utf-8", newline="") as f:
            text = f.read()
        nodes, edges, info = derive(entities, relations, polarity, text)
        write_ann(dest.with_suffix(".ann"), nodes, edges)

        stats["casos"] += 1
        stats["entidades originais"] += len(entities)
        stats["entidades mantidas"] += len(nodes)
        stats.update(info)
        for t, _, _ in nodes.values():
            stats[f"nó {t}"] += 1
        for label, _, _ in edges:
            stats[f"aresta {label}"] += 1
    if stats["casos"] == 0:
        raise FileNotFoundError(f"nenhum par .txt/.ann em {in_dir}")
    return stats


if __name__ == "__main__":
    from diplo_graph03 import config

    args = sys.argv[1:]
    if not args:
        src_dir, dst_dir = config.MACCROBAT_DIR, config.MACCROBAT_GRAFO_DIR
    elif len(args) == 2:
        src_dir, dst_dir = args
    else:
        sys.exit("uso: python -m diplo_graph03.preprocessing.label_mapping [PASTA_MACCROBAT PASTA_SAIDA]")
    print(f"{src_dir}  ->  {dst_dir}")
    report = convert_folder(src_dir, dst_dir)
    for k in sorted(report):
        print(f"{report[k]:6d}  {k}")