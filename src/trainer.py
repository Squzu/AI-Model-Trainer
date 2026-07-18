from pathlib import Path

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.config import load_config
from src.logger import get_logger

logger = get_logger()
config = load_config()


class LLMTrainer:

    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None):

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self):

        logger.info("Creating Training Arguments...")

        has_eval = self.eval_dataset is not None

        training_args = TrainingArguments(

            output_dir=config.training.output_dir,

            num_train_epochs=config.training.epochs,

            per_device_train_batch_size=config.training.batch_size,

            gradient_accumulation_steps=config.training.gradient_accumulation_steps,

            learning_rate=config.training.learning_rate,

            warmup_ratio=config.training.warmup_ratio,

            weight_decay=config.training.weight_decay,

            logging_steps=config.training.logging_steps,

            save_steps=config.training.save_steps,

            save_total_limit=config.training.save_total_limit,

            fp16=config.training.fp16,

            bf16=config.training.bf16,

            optim=config.training.optim,

            lr_scheduler_type=config.training.lr_scheduler,

            eval_strategy="steps" if has_eval else "no",

            eval_steps=config.training.save_steps if has_eval else None,

            report_to="tensorboard"

        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer_kwargs = dict(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # transformers >=4.49 renamed `tokenizer=` to `processing_class=`.
        # Try the new name first, fall back to the old one on older versions.
        try:
            trainer = Trainer(
                processing_class=self.tokenizer,
                **trainer_kwargs,
            )
        except TypeError:
            trainer = Trainer(
                tokenizer=self.tokenizer,
                **trainer_kwargs,
            )

        # Resume from the latest checkpoint if one exists, so an
        # interrupted run doesn't have to restart from scratch.
        output_dir = Path(config.training.output_dir)

        resume_from_checkpoint = None

        if output_dir.exists():
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if checkpoints:
                resume_from_checkpoint = str(checkpoints[-1])
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

        logger.info("Starting Training...")

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info("Saving Adapter...")

        self.model.save_pretrained(config.training.adapter_dir)

        self.tokenizer.save_pretrained(config.training.adapter_dir)

        logger.info("Training Completed Successfully")
