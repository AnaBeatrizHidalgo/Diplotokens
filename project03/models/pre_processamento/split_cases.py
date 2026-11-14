"""
Divisão treino / validação / teste POR CASO (documento).

Frases do mesmo caso nunca ficam em conjuntos diferentes, o que evita vazamento.
A divisão é salva em disco e reutilizada, então todos os classificadores são
avaliados exatamente nos mesmos casos.
"""

import json
import random
from pathlib import Path


def split_ids(ids, ratios=(0.7, 0.15, 0.15), seed=42) -> dict:
    """Embaralha (com semente fixa) e divide os ids; o teste fica com o que sobrar."""
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("as proporções devem somar 1")
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = round(ratios[0] * n), round(ratios[1] * n)
    return {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}


def save_splits(splits: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_splits(ids, path, ratios=(0.7, 0.15, 0.15), seed=42) -> dict:
    """Carrega a divisão salva (se bater com os casos atuais) ou cria e salva uma nova."""
    path = Path(path)
    ids = list(ids)
    if path.exists():
        splits = load_splits(path)
        if set(sum(splits.values(), [])) == set(ids):
            return splits
        print("aviso: splits.json não bate com os casos atuais; criando outro")
    splits = split_ids(ids, ratios, seed)
    save_splits(splits, path)
    return splits
