"""
Monta os datasets que os classificadores consomem, a partir dos embeddings.

Dois datasets, ambos no formato (X, y) que qualquer classificador estilo scikit-learn aceita:

  NÓS    : uma linha por subtoken     -> X = vetor do subtoken,  y = tag BIO ("B-Age", "I-Age", "O")
  ARESTAS: uma linha por par de nós   -> X = [v1 ; v2 ; |v1-v2| ; distância],  y = tipo da relação ou "NONE"

O BERT roda uma vez só: o resultado de cada divisão fica em cache (.npz).
`embedder` é qualquer objeto com `.name` e `.embed_text(text) -> (offsets, embeddings)`.
"""

from pathlib import Path

import numpy as np

NONE_LABEL = "NONE"


# ---------------------------------------------------------------- nós

def subtoken_tags(offsets, text, entities) -> list[str]:
    """Tag BIO de cada subtoken, comparando suas posições (caracteres) com as do .ann."""
    inside = [None] * len(text)
    starts = set()
    for ent in entities.values():
        for s, e in ent.spans:
            starts.add(s)
            for c in range(s, min(e, len(text))):
                inside[c] = ent.label            # entidades sobrepostas: vale a última

    tags = []
    for a, b in offsets:
        labs = [inside[c] for c in range(a, b) if inside[c] is not None]
        if not labs:
            tags.append("O")
        else:
            prefix = "B" if any(c in starts for c in range(a, b)) else "I"
            tags.append(f"{prefix}-{labs[0]}")
    return tags


def build_token_dataset(docs, splits, embedder, out_dir, force=False) -> None:
    """Gera (ou reaproveita) um arquivo .npz por divisão com X, y, doc e offsets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {d.id: d for d in docs}

    for split, ids in splits.items():
        path = out_dir / f"{split}_{embedder.name}.npz"
        if path.exists() and not force:
            print(f"[{split}] cache encontrado: {path.name}")
            continue
        X, y, doc_ids, offs = [], [], [], []
        for doc_id in ids:
            d = by_id[doc_id]
            offsets, emb = embedder.embed_text(d.text)
            X.append(emb)
            offs.append(offsets)
            y += subtoken_tags(offsets, d.text, d.entities)
            doc_ids += [doc_id] * len(offsets)
        np.savez_compressed(path, X=np.concatenate(X), y=np.array(y),
                            doc=np.array(doc_ids), offsets=np.concatenate(offs))
        print(f"[{split}] {len(ids)} casos -> {path.name}")


def load_token_dataset(out_dir, split, embedder_name) -> dict:
    """Carrega um .npz: dict com X, y, doc (id do caso) e offsets (posições em caracteres)."""
    z = np.load(Path(out_dir) / f"{split}_{embedder_name}.npz", allow_pickle=False)
    return {k: z[k] for k in ("X", "y", "doc", "offsets")}


# ---------------------------------------------------------------- arestas

def doc_slices(doc_array) -> dict:
    """Intervalo (slice) de cada caso dentro dos arrays do dataset de nós."""
    if len(doc_array) == 0:
        return {}
    bounds = np.flatnonzero(doc_array[1:] != doc_array[:-1]) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(doc_array)]
    return {str(doc_array[s]): slice(s, e) for s, e in zip(starts, ends)}


def entity_vector(offsets, emb, spans):
    """Vetor de um nó: média dos subtokens que se sobrepõem às posições da entidade."""
    mask = np.zeros(len(offsets), bool)
    for s, e in spans:
        mask |= (offsets[:, 0] < e) & (offsets[:, 1] > s)
    return emb[mask].mean(axis=0) if mask.any() else None


def build_edge_dataset(docs, token_data, max_char_dist=300, neg_ratio=None, seed=42) -> dict:
    """
    Exemplos para a classificação de arestas.

    - positivos: toda relação do .ann (direcionada: arg1 -> arg2);
    - negativos ("NONE"): pares de entidades a até `max_char_dist` caracteres sem relação;
    - neg_ratio: se informado (ex.: 5), mantém só 5 negativos por positivo. Use no treino
      para lidar com o desbalanceamento; deixe None em validação e teste.
    Retorna dict com X, y e meta (caso, id do nó 1, id do nó 2).
    """
    by_id = {d.id: d for d in docs}
    slices = doc_slices(token_data["doc"])
    rng = np.random.default_rng(seed)

    vec_cache = {}

    def vec(doc_id, eid):
        if (doc_id, eid) not in vec_cache:
            sl, d = slices[doc_id], by_id[doc_id]
            vec_cache[(doc_id, eid)] = entity_vector(
                token_data["offsets"][sl], token_data["X"][sl], d.entities[eid].spans)
        return vec_cache[(doc_id, eid)]

    # 1) lista os pares (barato, ainda sem vetores)
    pairs = []
    for doc_id in slices:
        d = by_id[doc_id]
        rel = {(r.arg1, r.arg2): r.label for r in d.relations
               if r.arg1 in d.entities and r.arg2 in d.entities}
        ids = [e for e in d.entities if vec(doc_id, e) is not None]
        pos0 = {e: min(s for s, _ in d.entities[e].spans) for e in ids}
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                dist = abs(pos0[a] - pos0[b])
                label = rel.get((a, b))
                if label is None:
                    if dist > max_char_dist:
                        continue
                    label = NONE_LABEL
                pairs.append((doc_id, a, b, label, dist))

    # 2) subamostra negativos, se pedido
    if neg_ratio is not None:
        pos = [p for p in pairs if p[3] != NONE_LABEL]
        neg = [p for p in pairs if p[3] == NONE_LABEL]
        keep = rng.choice(len(neg), size=min(len(neg), neg_ratio * len(pos)), replace=False)
        pairs = pos + [neg[i] for i in sorted(keep)]

    # 3) só agora calcula os vetores
    X, y, meta = [], [], []
    for doc_id, a, b, label, dist in pairs:
        va, vb = vec(doc_id, a), vec(doc_id, b)
        X.append(np.concatenate([va, vb, np.abs(va - vb),
                                 [min(dist, max_char_dist) / max_char_dist]]))
        y.append(label)
        meta.append((doc_id, a, b))
    dim = 3 * token_data["X"].shape[1] + 1
    return {"X": np.array(X, dtype=np.float32).reshape(len(X), dim),
            "y": np.array(y), "meta": np.array(meta).reshape(len(meta), 3)}
