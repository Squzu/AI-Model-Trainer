from src.model import load_model

model, tokenizer = load_model()

print()

print("=" * 50)

print("Model Loaded Successfully")

print(model.__class__.__name__)

print(tokenizer.__class__.__name__)

print("=" * 50)
