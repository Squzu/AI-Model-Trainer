import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from pathlib import Path

from src.config import load_config
from src.logger import get_logger

logger = get_logger()
config = load_config()


RAW_DATASET = Path("datasets/raw/train.json")

TRAIN_OUTPUT = Path(config.dataset.train_file)

VALID_OUTPUT = Path(config.dataset.validation_file)


def validate_record(record):

    required = ["instruction", "input", "output"]

    for field in required:
        if field not in record:
            raise ValueError(f"Missing field: {field}")


def build_prompt(record):

    instruction = record["instruction"].strip()
    input_text = record["input"].strip()
    output = record["output"].strip()

    if input_text:

        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )

    else:

        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    return {"text": prompt}


def main():

    logger.info("Loading raw dataset...")

    with open(RAW_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    for item in data:

        validate_record(item)

        processed.append(build_prompt(item))

    random.shuffle(processed)

    split = int(len(processed) * (1 - config.dataset.validation_split))

    train = processed[:split]

    validation = processed[split:]

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)

    with open(VALID_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    logger.info(f"Training Samples : {len(train)}")
    logger.info(f"Validation Samples : {len(validation)}")
    logger.info("Dataset preparation completed.")


if __name__ == "__main__":
    main()
