import torch
from config import TrainConfig

def get_optimizer(config: TrainConfig, net: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer : torch.optim.Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, config.beta_2))
    elif config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise Exception("Invalid optimizer")

    return optimizer
