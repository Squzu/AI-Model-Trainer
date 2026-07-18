import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.exporter import merge_lora
from src.logger import get_logger


logger = get_logger()


def main():

    logger.info("=" * 60)
    logger.info("LoRA Merge")
    logger.info("=" * 60)

    merge_lora()


if __name__ == "__main__":
    main()
