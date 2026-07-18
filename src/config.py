from pathlib import Path
import yaml
from box import Box


class ConfigLoader:
    """
    Loads all YAML configuration files and merges them into
    one configuration object.
    """

    def __init__(self, config_dir="configs"):

        self.config_dir = Path(config_dir)

        self.config = Box()

        self.load()

    def read_yaml(self, filename):

        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Missing configuration file: {filepath}")

        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def load(self):

        model = self.read_yaml("model.yaml")
        training = self.read_yaml("training.yaml")
        dataset = self.read_yaml("dataset.yaml")
        inference = self.read_yaml("inference.yaml")

        self.config.merge_update(model)
        self.config.merge_update(training)
        self.config.merge_update(dataset)
        self.config.merge_update(inference)

    def get(self):
        return self.config


def load_config():

    return ConfigLoader().get()
