"""
Copia todos os arquivos .txt dos casos de TESTE (definidos em splits.json)
para uma pasta dedicada dentro de data/interim/.

Uso (a partir da raiz project03):
    python -m diplo_graph03.preprocessing.export_test_texts
"""

import shutil
from pathlib import Path
from diplo_graph03 import config
from diplo_graph03.preprocessing.dataset import load_splits


def find_txt_file(doc_id: str, text_root: Path) -> Path:
    """Localiza o caminho do arquivo <doc_id>.txt correspondente ao caso clínico."""
    direct_path = text_root / f"{doc_id}.txt"
    if direct_path.exists():
        return direct_path

    matches = list(text_root.rglob(f"{doc_id}.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Arquivo .txt não encontrado para o caso {doc_id} em {text_root}")


def export_test_texts(out_dir: Path, source_dir: Path) -> None:
    splits = load_splits(config.SPLITS_JSON)
    test_ids = splits["test"]

    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for doc_id in test_ids:
        src_txt = find_txt_file(doc_id, source_dir)
        dst_txt = out_dir / f"{doc_id}.txt"
        shutil.copyfile(src_txt, dst_txt)  # cópia byte a byte preservando offsets
        copied += 1

    print(f"Exportados {copied} casos de teste (.txt) para: {out_dir}")


if __name__ == "__main__":
    test_texts_folder = config.INTERIM_DATA_DIR / "maccrobat_test_texts"
    export_test_texts(
        out_dir=test_texts_folder,
        source_dir=config.MACCROBAT_GRAFO_DIR,
    )