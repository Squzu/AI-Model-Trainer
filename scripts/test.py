import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_inference_model, generate_response
from src.logger import get_logger


logger = get_logger()


def main():

    logger.info("=" * 60)
    logger.info("LLM Inference")
    logger.info("=" * 60)

    model, tokenizer = load_inference_model()

    print("\nType 'exit' to quit.\n")

    while True:

        question = input("Question > ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        print("\nGenerating...\n")

        answer = generate_response(
            model,
            tokenizer,
            question,
        )

        print("=" * 60)
        print(answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
