import copy
import time
import torch
import wandb
import argparse

import torch.nn as nn
import torch.nn.functional as F

from models import get_net
from dataset import get_loaders_geoGuessr
from utils import TrainConfig, load_config, get_optimizer

EARTH_RADIUS = 6371000  # meter

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Haversine distance loss and geoguessr score
def loss_fn(pred, target):
    pred_lon, pred_lat = pred[:, 0], pred[:, 1]
    true_lon, true_lat = target[:, 0], target[:, 1]

    pred_lon = torch.deg2rad(pred_lon)
    pred_lat = torch.deg2rad(pred_lat)
    true_lon = torch.deg2rad(true_lon)
    true_lat = torch.deg2rad(true_lat)

    delta_phi = true_lat - pred_lat
    delta_lambda = true_lon - pred_lon

    a = torch.sin(delta_phi / 2) ** 2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(delta_lambda / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = EARTH_RADIUS * c / 1000

    with torch.no_grad():
        scaling_factor = 2000  # km
        score = 5000 * torch.exp(-distance / scaling_factor)

    return distance.mean(), score.mean()


def evaluate(net, loader):
    val_distance = 0.0
    val_score = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = net(X)  # BxCx2
            distance, score = loss_fn(out, y)
            bs = X.size(0)

            val_distance += distance.item() * bs
            val_score += score.item() * bs
            total_samples += bs
    val_distance /= total_samples
    val_score /= total_samples

    return val_distance, val_score


def train(
    config: TrainConfig,
    net,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader
):
    best_distance = float('inf')
    best_net = None
    early_stop_counter = 0
    global_step = 0

    # Evaluate
    start = time.perf_counter()
    net.eval()
    val_distance, val_score = evaluate(net, eval_loader)
    net.train()
    taken = time.perf_counter() - start
    wandb.log(
        {
            "epoch": 0,
            "train_examples": global_step,
            "eval_distance": val_distance,
            "eval_score": val_score,
        }
    )
    print(
        f"[Eval] Epoch 0,",
        f"Avg score: {val_score:,.2f}, Avg distance: {val_distance:,.2f}",
        f"Time Taken: {taken:.2f}s",
    )

    start = time.perf_counter()
    for e in range(config.epochs):
        running_distance = 0.
        running_score = 0.
        net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            out = net(X)  # Bx2

            distance, score = loss_fn(out, y)
            distance.backward()

            running_distance += distance.item()
            running_score += score.item()

            if config.gradient_clipping_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config.gradient_clipping_norm
                )

            optimizer.step()
            optimizer.zero_grad()
            global_step += bs

            if (i+1) % config.log_interval == 0:
                taken = time.perf_counter() - start
                avg_distance = running_distance / config.log_interval
                avg_score = running_score / config.log_interval
                ips = config.log_interval / taken

                wandb.log(
                    {
                        "epoch": e,
                        "batch": i,
                        "train_examples": global_step,
                        "train_score": avg_score,
                        "train_distance": avg_distance,
                    }
                )
                print(
                    f"Epoch {e}, step {i} (global step {global_step}),",
                    f"Avg score: {avg_score:,.2f}, Avg distance: {avg_distance:,.2f}",
                    f"Time Taken: {taken:.2f}s, ({ips:.2f} i/s)",
                )

                running_distance = 0.
                running_score = 0.
                start = time.perf_counter()

        net.eval()
        val_distance, val_score = evaluate(net, eval_loader)
        net.train()
        taken = time.perf_counter() - start
        wandb.log(
            {
                "epoch": e+1,
                "train_examples": global_step,
                "eval_distance": val_distance,
                "eval_score": val_score,
            }
        )
        print(
            f"[Eval] Epoch {e+1},",
            f"Avg score: {val_score:,.2f}, Avg distance: {val_distance:,.2f}",
            f"Time Taken: {taken:.2f}s",
        )

        # Check for early stop
        if val_distance < best_distance:
            best_distance = val_distance
            best_net = copy.deepcopy(net)
        else:
            early_stop_counter += 1
            if early_stop_counter > config.early_stop:
                net = best_net
                break

    net.eval()
    return evaluate(net, test_loader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to train config", required=True
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model before training"
    )
    return parser.parse_args()


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
        net = get_net(num_classes=train_config.num_classes, freeze_weights=train_config.freeze_weights, net_name=train_config.net_name, device=device)
        optimizer = get_optimizer(train_config, net)

        if args.compile:
            print("Compiling network")
            net = torch.compile(net)

        num_params = sum(p.numel() for p in net.parameters())
        print(f"Model parameters {num_params:,}")
        config_dict["num_params"] = num_params
        wandb.config.update(config_dict, allow_val_change=True)

        train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
            train_config.batch_size, train_config.num_classes, directory=train_config.dataset_dir
        )

        print("Training on device:", device)
        test_loss, test_distance, test_score = train(
            config=train_config,
            net=net,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader
        )
        wandb.log(
            {
                "test_loss": test_loss,
                "test_distance": test_distance,
                "test_score": test_score,
            }
        )

    sweep_id = wandb.sweep(sweep_config, project="GeoGuessrCoordinates")
    wandb.agent(sweep_id, function=sweep_train)
