import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.exporter import (
    export_huggingface,
    export_gguf,
    export_ollama,
)

from src.logger import get_logger

logger = get_logger()


def main():

    parser = argparse.ArgumentParser(
        description="Export trained models."
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "hf",
            "gguf",
            "ollama",
        ],
        help="Export target",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Model Export")
    logger.info("=" * 60)

    if args.target == "hf":
        export_huggingface()

    elif args.target == "gguf":
        export_gguf()

    elif args.target == "ollama":
        export_ollama()


if __name__ == "__main__":
    main()
