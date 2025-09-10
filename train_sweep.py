import torch
import wandb

from models import get_net
from train import get_args, train
from dataset import get_loaders_geoGuessr
from utils import load_config, get_optimizer


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if __name__ == "__main__":
    args = get_args()
    train_config = load_config(args.config)

    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "test_distance", "goal": "minimize"},
        "parameters": {
            "optimizer": {"values": ["adam", "adamW", "sgd"]},
            "beta_2": {"values": [0.95, 0.97, 0.99, 0.999]},
            "learning_rate": {"min": 8e-5, "max": 5e-4},
            "weight_decay": {"values": [0.0, 0.01, 0.05]},
        },
    }

    def sweep_train():
        wandb.init(project="GeoGuessrCoordinates", name=train_config.run_name)
        config = wandb.config

        # update configs with sweep params
        train_config.optimizer = config.optimizer
        train_config.beta_2 = config.beta_2
        train_config.learning_rate = config.learning_rate
        train_config.weight_decay = config.weight_decay

        config_dict = {**vars(train_config)}

        print("Setting up model")
        net = get_net(freeze_weights=train_config.freeze_weights, net_name=train_config.net_name, device=device)
        optimizer = get_optimizer(train_config, net)

        if args.compile:
            print("Compiling network")
            net = torch.compile(net)

        num_params = sum(p.numel() for p in net.parameters())
        print(f"Model parameters {num_params:,}")
        config_dict["num_params"] = num_params
        wandb.config.update(config_dict, allow_val_change=True)

        train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
            train_config.batch_size, directory=train_config.dataset_dir
        )

        print("Training on device:", device)
        test_distance, test_score = train(
            config=train_config,
            net=net,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader
        )
        wandb.log(
            {
                "test_distance": test_distance,
                "test_score": test_score,
            }
        )

    sweep_id = wandb.sweep(sweep_config, project="GeoGuessrCoordinates")
    wandb.agent(sweep_id, function=sweep_train)
