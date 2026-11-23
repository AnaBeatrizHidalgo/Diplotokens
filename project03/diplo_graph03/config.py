from pathlib import Path

RANDOM_SEED = 42

# paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# MultiCaRe (amostra)
SAMPLE_DATA_DIR = RAW_DATA_DIR / "sample"

CASES_CSV = SAMPLE_DATA_DIR / "cases.csv"
DATA_DICT_CSV = SAMPLE_DATA_DIR / "data_dictionary.csv"
METADATA_CSV = SAMPLE_DATA_DIR / "metadata.csv"

RESULTS_DIR = DATA_DIR / "results"

# artefatos e experimentos
MODELS_DIR = PROJ_ROOT / "models"
CONFIGS_DIR = PROJ_ROOT / "configs"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MACCROBAT2020
MACCROBAT_DIR = EXTERNAL_DATA_DIR / "MACCROBAT2020"                # original, intocado
MACCROBAT_GRAFO_DIR = INTERIM_DATA_DIR / "maccrobat_grafo"         # saída do label_mapping
SPLITS_JSON = CONFIGS_DIR / "splits.json"             # divisão treino/val/teste por caso
EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "embeddings"
WINDOWS_DIR = PROCESSED_DATA_DIR / "windows"