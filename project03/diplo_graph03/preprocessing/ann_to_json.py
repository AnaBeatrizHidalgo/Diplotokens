import json
from pathlib import Path
from diplo_graph03 import config
from diplo_graph03.preprocessing.dataset import load_docs, load_splits
from diplo_graph03.schema import Graph

SPLIT_TO_FOLDER = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def export_graphs_by_split(out_dir: Path) -> None:
    splits = load_splits(config.SPLITS_JSON)

    id_to_folder = {
        doc_id: SPLIT_TO_FOLDER[split_key]
        for split_key, ids in splits.items()
        for doc_id in ids
    }

    for folder_name in SPLIT_TO_FOLDER.values():
        (out_dir / folder_name).mkdir(parents=True, exist_ok=True)

    docs = load_docs(config.MACCROBAT_GRAFO_DIR)
    counts = {folder: 0 for folder in SPLIT_TO_FOLDER.values()}

    for doc in docs:
        folder_name = id_to_folder.get(doc.id)
        if folder_name is None:
            print(f"Warning: {doc.id} not found in splits.json, skipping.")
            continue

        graph = Graph.from_brat(doc.entities, doc.relations, doc_id=doc.id)
        issues = graph.problems()
        if issues:
            print(f"Warning in {doc.id}: {issues}")

        # Remove "doc_id" so the JSON only contains "nodes" and "edges"
        graph_dict = graph.to_dict()
        graph_dict.pop("doc_id", None)

        out_path = out_dir / folder_name / f"{doc.id}.json"
        out_path.write_text(
            json.dumps(graph_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        counts[folder_name] += 1

    print(f"Exported JSON graphs (without doc_id) to {out_dir}:")
    for folder_name, count in counts.items():
        print(f"  {folder_name:<10}: {count} files")


if __name__ == "__main__":
    export_graphs_by_split(config.INTERIM_DATA_DIR / "maccrobat_grafo_json")