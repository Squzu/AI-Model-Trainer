from src.config import load_config

config = load_config()

print(config.model.name)
print(config.training.batch_size)
print(config.lora.r)
print(config.dataset.train_file)
print(config.inference.temperature)
