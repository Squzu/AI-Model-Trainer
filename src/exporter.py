from pathlib import Path
import shutil
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import load_config
from src.logger import get_logger


logger = get_logger()
config = load_config()


MERGED_MODEL_DIR = Path("models/merged")
EXPORT_DIR = Path("exports")


def _check_merged_model():

    if not MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(
            "Merged model not found. Run scripts/merge_lora.py first."
        )


def merge_lora():
    """
    Merge the trained LoRA adapter into the base model and save
    the result to models/merged.

    This is done in fp16 on CPU rather than on the GPU. bitsandbytes
    4-bit weights cannot be merged directly (they'd need to be
    de-quantized first), and doing the merge on CPU avoids needing
    extra VRAM for this one-time step on small GPUs (e.g. 4GB cards).
    It's slower than a GPU merge, but it only runs once per adapter.
    """

    adapter_dir = Path(config.training.adapter_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"No trained adapter found at {adapter_dir}. "
            "Run scripts/train.py first."
        )

    logger.info(f"Loading base model in fp16 on CPU: {config.model.name}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
        )
    except Exception as e:
        logger.error(f"Failed to load base model '{config.model.name}': {e}")
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=config.model.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading LoRA adapter from: {adapter_dir}")

    merged_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
    )

    logger.info("Merging LoRA weights into base model...")

    merged_model = merged_model.merge_and_unload()

    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {MERGED_MODEL_DIR}")

    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    logger.info("Merge completed successfully.")


def export_huggingface():

    logger.info("Exporting Hugging Face model...")

    _check_merged_model()

    output_dir = EXPORT_DIR / "huggingface"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(
        MERGED_MODEL_DIR,
        output_dir,
    )

    logger.info(f"Export completed: {output_dir}")


def export_gguf():

    logger.info("Exporting GGUF model...")

    _check_merged_model()

    output_dir = EXPORT_DIR / "gguf"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    converter = (
        Path("tools")
        / "llama.cpp"
        / "convert_hf_to_gguf.py"
    )

    if not converter.exists():
        logger.error(
            "convert_hf_to_gguf.py not found. Clone llama.cpp into "
            "tools/llama.cpp first: "
            "git clone https://github.com/ggerganov/llama.cpp tools/llama.cpp"
        )
        return

    output_file = output_dir / "model.gguf"

    command = [
        "python3",
        str(converter),
        str(MERGED_MODEL_DIR),
        "--outfile",
        str(output_file),
        "--outtype",
        "q8_0",
    ]

    logger.info("Running GGUF conversion...")

    result = subprocess.run(command)

    if result.returncode != 0:
        logger.error("GGUF conversion failed.")
        return

    logger.info(f"GGUF model created: {output_file}")


def export_ollama():

    logger.info("Preparing Ollama export...")

    _check_merged_model()

    if shutil.which("ollama") is None:
        logger.error("Ollama is not installed.")
        return

    output_dir = EXPORT_DIR / "ollama"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    modelfile = output_dir / "Modelfile"

    gguf_path = (EXPORT_DIR / "gguf" / "model.gguf").resolve()

    if not gguf_path.exists():
        logger.error(
            "GGUF model not found. Run export.py --target gguf first."
        )
        return

    modelfile.write_text(
        f"""FROM {gguf_path}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    )

    logger.info(f"Generated Modelfile: {modelfile}")
    logger.info(
        "Run: ollama create ai-model-trainer -f exports/ollama/Modelfile"
    )
