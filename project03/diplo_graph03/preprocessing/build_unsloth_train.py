"""
Gera o dataset de fine-tuning no formato Unsloth (instruction, input, output)
EXCLUSIVAMENTE para os 140 casos de treino definidos em splits.json.

Uso (a partir da raiz project03):
    python -m diplo_graph03.preprocessing.build_unsloth_train
"""

import json
from pathlib import Path
from diplo_graph03 import config
from diplo_graph03.preprocessing.dataset import load_splits

INSTRUCTION = (
    "You are an AI Engineer specialized in clinical Natural Language Processing and Knowledge Graph "
    "extraction. Extract a clinical knowledge graph from the given medical case report and output ONLY "
    "a valid JSON object with two keys: 'nodes' and 'edges' (no markdown, no explanations).\n\n"
    "Schema rules:\n"
    "- 'nodes': list of objects with keys 'type', 'spans' (list of [start, end] character offsets in the text), "
    "and 'text'. Allowed node types: PATIENT, DISEASE, SYMPTOM, DRUG, EXAM, ANATOMY, EFFECT, RESULT.\n"
    "- 'edges': list of objects with keys 'type', 'src' (0-based source node index), and 'dst' (0-based target node index). "
    "Allowed edge types and directions:\n"
    "  * PATIENT -> DISEASE: FAMILY_HISTORY | PERSONAL_HISTORY | DIAGNOSED_WITH\n"
    "  * PATIENT -> SYMPTOM: HAS_SYMPTOM\n"
    "  * PATIENT -> DRUG: TAKES\n"
    "  * PATIENT -> EXAM: UNDERGOES\n"
    "  * SYMPTOM -> ANATOMY: OCCURS_IN\n"
    "  * DISEASE -> ANATOMY: AFFECTS\n"
    "  * EXAM -> RESULT: REVEALS\n"
    "  * DRUG -> EFFECT: HAS_EFFECT"
)


def find_clinical_text(doc_id: str, text_root: Path) -> str:
    """Localiza o arquivo <doc_id>.txt correspondente ao caso clínico."""
    direct_path = text_root / f"{doc_id}.txt"
    if direct_path.exists():
        return direct_path.read_text(encoding="utf-8")

    # Caso os .txt estejam organizados em subpastas dentro de text_root
    matches = list(text_root.rglob(f"{doc_id}.txt"))
    if matches:
        return matches[0].read_text(encoding="utf-8")

    raise FileNotFoundError(f"Arquivo de texto não encontrado para o caso {doc_id} em {text_root}")


def build_unsloth_train_dataset(
    train_json_dir: Path,
    text_dir: Path,
    output_file: Path,
    compact_output_json: bool = True,
) -> None:
    splits = load_splits(config.SPLITS_JSON)
    train_ids = splits["train"]

    dataset_rows = []
    for doc_id in train_ids:
        graph_path = train_json_dir / f"{doc_id}.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Grafo JSON não encontrado: {graph_path}")

        clinical_text = find_clinical_text(doc_id, text_dir)
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))

        # Serializa o grafo como string JSON para o campo 'output' do Unsloth
        output_str = (
            json.dumps(graph_data, ensure_ascii=False, separators=(",", ":"))
            if compact_output_json
            else json.dumps(graph_data, ensure_ascii=False, indent=2)
        )

        dataset_rows.append({
            "instruction": INSTRUCTION,
            "input": clinical_text,
            "output": output_str,
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(dataset_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset Unsloth gerado com sucesso em: {output_file}")
    print(f"Total de linhas (apenas treino): {len(dataset_rows)}")


if __name__ == "__main__":
    train_graphs_folder = config.INTERIM_DATA_DIR / "maccrobat_grafo_json" / "train"
    unsloth_output_path = config.PROCESSED_DATA_DIR / "unsloth_train.json"

    build_unsloth_train_dataset(
        train_json_dir=train_graphs_folder,
        text_dir=config.MACCROBAT_GRAFO_DIR,
        output_file=unsloth_output_path,
        compact_output_json=True,  # economiza tokens no treinamento do LLM
    )