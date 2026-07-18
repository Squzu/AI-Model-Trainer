from datasets import load_dataset

from src.config import load_config
from src.logger import get_logger


logger = get_logger()
config = load_config()


def load_training_dataset():
    """
    Load the processed training dataset.
    """

    logger.info("Loading processed training dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": config.dataset.train_file,
            "validation": config.dataset.validation_file,
        },
    )

    logger.info(f"Training samples   : {len(dataset['train'])}")
    logger.info(f"Validation samples : {len(dataset['validation'])}")

    return dataset


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the processed dataset.
    """

    logger.info("Tokenizing dataset...")

    max_length = config.training.max_seq_length

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Labels for causal language modeling
        tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    tokenized = dataset.map(
        tokenize,
        batched=False,
        remove_columns=["text"],
    )

    logger.info("Tokenization completed.")

    return tokenized
