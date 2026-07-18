import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import platform
import shutil
import subprocess

from src.logger import get_logger
from src.paths import LLAMA_CPP

logger = get_logger()


def check_command(command):
    return shutil.which(command) is not None


def check_python():
    logger.info(f"Python : {platform.python_version()}")


def check_git():
    logger.info(f"Git    : {'OK' if check_command('git') else 'NOT FOUND'}")


def check_gpp():
    logger.info(f"G++    : {'OK' if check_command('g++') else 'NOT FOUND'}")


def check_cmake():
    logger.info(f"CMake  : {'OK' if check_command('cmake') else 'NOT FOUND'}")


def check_cuda():

    if not check_command("nvidia-smi"):
        logger.warning("CUDA GPU : NOT FOUND")
        return

    try:

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()

        logger.info(f"GPU    : {output}")

    except Exception:
        logger.warning("Unable to query GPU.")


def check_ollama():

    if not check_command("ollama"):
        logger.warning("Ollama : NOT INSTALLED")
        return

    version = subprocess.check_output(
        ["ollama", "--version"],
        text=True,
    ).strip()

    logger.info(f"Ollama : {version}")


def check_llamacpp():

    if LLAMA_CPP.exists():
        logger.info("llama.cpp : FOUND")
    else:
        logger.warning("llama.cpp : NOT INSTALLED")


def main():

    logger.info("=" * 60)
    logger.info("AI Model Trainer Setup Check")
    logger.info("=" * 60)

    check_python()
    check_git()
    check_gpp()
    check_cmake()
    check_cuda()
    check_ollama()
    check_llamacpp()

    logger.info("=" * 60)
    logger.info("Setup Check Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
