from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import torch

from src.config import load_config
from src.logger import get_logger


logger = get_logger()
config = load_config()


def load_model():
    """
    Load the base model and apply LoRA.
    """

    logger.info("Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
        )
    except Exception as e:
        logger.error(
            f"Could not load tokenizer for '{config.model.name}'. "
            f"Check the model name in configs/model.yaml and your "
            f"internet connection. Original error: {e}"
        )
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {config.model.name}")

    if config.model.load_in_4bit:

        logger.info("Using 4-bit quantization")

        compute_dtype = getattr(
            torch,
            config.model.bnb_4bit_compute_dtype,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=config.model.trust_remote_code,
                cache_dir=config.model.cache_dir,
            )
        except Exception as e:
            logger.error(
                f"Could not load base model '{config.model.name}' in "
                f"4-bit. Check the model name, your internet connection, "
                f"and that bitsandbytes is installed correctly. "
                f"Original error: {e}"
            )
            raise

        model = prepare_model_for_kbit_training(model)

    else:

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=config.model.trust_remote_code,
                cache_dir=config.model.cache_dir,
            )
        except Exception as e:
            logger.error(
                f"Could not load base model '{config.model.name}'. "
                f"Check the model name and your internet connection. "
                f"Original error: {e}"
            )
            raise

    # ---------------------------------
    # Memory Optimizations
    # ---------------------------------

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    # ---------------------------------
    # LoRA Configuration
    # ---------------------------------

    logger.info("Applying LoRA")

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer
