"""
Gera e valida o arquivo splits.json (divisão treino / validação / teste por caso).

Uso (a partir da raiz project03):
    python -m diplo_graph03.preprocessing.make_splits
"""

from diplo_graph03 import config
from diplo_graph03.preprocessing.dataset import get_splits, load_docs


def main() -> None:
    # Carrega os casos já convertidos para o esquema do grafo
    docs = load_docs(config.MACCROBAT_GRAFO_DIR)
    doc_ids = [d.id for d in docs]

    # Gera splits.json (ou carrega o existente se já bater com os IDs)
    splits = get_splits(
        ids=doc_ids,
        path=config.SPLITS_JSON,
        ratios=(0.7, 0.15, 0.15),
        seed=config.RANDOM_SEED,
    )

    print(f"Arquivo salvo em: {config.SPLITS_JSON}")
    print(f"Total de casos: {len(doc_ids)}")
    for split_name, ids in splits.items():
        print(f"  {split_name:<5}: {len(ids)} casos (primeiros 3: {ids[:3]})")


if __name__ == "__main__":
    main()