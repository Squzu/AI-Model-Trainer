from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel

import torch

from src.config import load_config
from src.logger import get_logger


logger = get_logger()
config = load_config()


def load_inference_model():

    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=config.model.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")

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

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=config.model.cache_dir,
    )

    logger.info("Loading LoRA Adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        config.inference.adapter_path,
    )

    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, question):

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            top_k=config.inference.top_k,
            do_sample=config.inference.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return response.split("### Response:")[-1].strip()
