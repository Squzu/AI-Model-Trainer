from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = PROJECT_ROOT / "configs"
DATASETS = PROJECT_ROOT / "datasets"
RAW_DATASETS = DATASETS / "raw"
PROCESSED_DATASETS = DATASETS / "processed"

MODELS = PROJECT_ROOT / "models"
BASE_MODELS = MODELS / "base"
ADAPTERS = MODELS / "adapters"
MERGED_MODEL = MODELS / "merged"

EXPORTS = PROJECT_ROOT / "exports"
LOGS = PROJECT_ROOT / "logs"
OUTPUTS = PROJECT_ROOT / "outputs"

TOOLS = PROJECT_ROOT / "tools"
LLAMA_CPP = TOOLS / "llama.cpp"

SCRIPTS = PROJECT_ROOT / "scripts"
SRC = PROJECT_ROOT / "src"
