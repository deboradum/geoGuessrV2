import yaml
from dataclasses import dataclass


@dataclass
class EmbedModelConfig:
    size: str = "b16"
    embedding_dim: int = 256
    freeze_weights: bool = False

@dataclass
class RefinementModelConfig:
    size: str = "b16"

@dataclass
class TrainConfig:
    dataset_dir: str = "dataset/"
    log_interval: int = 100
    seed: int = 123
    epochs: int = 2
    optimizer: str = "adamW"
    beta_2: float = 0.95
    learning_rate: float = 0.0001
    # learning_rate_warmup: int
    weight_decay: float = 0.05
    batch_size: int = 64
    gradient_clipping_norm: float = 1.0
    early_stop: int = 3
    run_name: str = "You forgot to change the run name"

@dataclass
class Config:
    embedModelConfig: EmbedModelConfig
    refinementModelConfig: RefinementModelConfig
    trainConfig : TrainConfig
    device: str = "cuda"


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
