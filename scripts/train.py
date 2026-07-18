import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.model import load_model
from src.dataset import load_training_dataset, tokenize_dataset
from src.trainer import LLMTrainer

logger = get_logger()


def main():

    logger.info("=" * 60)
    logger.info("LLM Fine-Tuning Framework")
    logger.info("=" * 60)

    logger.info("Loading Model...")
    model, tokenizer = load_model()

    logger.info("Loading Dataset...")
    dataset = load_training_dataset()
    dataset = tokenize_dataset(dataset, tokenizer)

    logger.info("Initializing Trainer...")
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
    )

    trainer.train()

    logger.info("Training Finished Successfully!")


if __name__ == "__main__":
    main()
