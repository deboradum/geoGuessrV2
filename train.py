import time
import torch
import wandb
import argparse

import torch.nn.functional as F

from tqdm import tqdm
from models import get_net
from dataset import get_loaders_googleMaps, get_loaders_geoGuessr

EARTH_RADIUS = 6371000
MAX_DISTANCE = 20000000.0

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def loss_fn(pred, truth):
    loss = F.mse_loss(pred, truth, reduction="none").mean(dim=1).mean()

    return loss


# Gets average haversine distance
def get_distance(pred, truth):
    phi_pred = torch.deg2rad(pred[:, 0])  # B, 1
    phi_truth = torch.deg2rad(truth[:, 0])  # B, 1
    delta_phi = torch.deg2rad(truth[:, 0] - pred[:, 0])  # B, 1
    delta_lambda = torch.deg2rad(truth[:, 1] - pred[:, 1])  # B, 1
    a = (
       torch.sin(delta_phi / 2.0) ** 2
       + torch.cos(phi_pred)
       * torch.cos(phi_truth)
       * torch.sin(delta_lambda / 2.0) ** 2
    )  # B, 1
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # B, 1
    distance = EARTH_RADIUS * c  # B, 1
    distance /= 1000  # Distance in m > km

    return distance.mean()


# Gets average geoguessr score (100 meters = 4999 points)
def get_geoguessr_score(pred, truth, scaling_factor=2):
    phi_pred = torch.deg2rad(pred[:, 0])  # B, 1
    phi_truth = torch.deg2rad(truth[:, 0])  # B, 1
    delta_phi = torch.deg2rad(truth[:, 0] - pred[:, 0])  # B, 1
    delta_lambda = torch.deg2rad(truth[:, 1] - pred[:, 1])  # B, 1
    a = (
        torch.sin(delta_phi / 2.0) ** 2
        + torch.cos(phi_pred)
        * torch.cos(phi_truth)
        * torch.sin(delta_lambda / 2.0) ** 2
    )  # B, 1
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # B, 1
    distance = EARTH_RADIUS * c  # B, 1
    scaling_factor = 2000000
    score = 5000 * torch.exp(-distance / scaling_factor)

    return torch.mean(score)


def evaluate(net, loader, loss_fn):
    net.eval()
    val_loss = 0.0
    val_distance = 0.0
    val_score = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = net(X)
            out_lat = out[:, 0] * 90  # Scale latitude to [-90, 90]
            out_lon = out[:, 1] * 180  # Scale longitude to [-180, 180]
            out_scaled = torch.stack([out_lat, out_lon], dim=1)

            loss = loss_fn(out_scaled, y)
            distance = get_distance(out_scaled, y)
            score = get_geoguessr_score(out_scaled, y)
            val_loss += loss.item()
            val_distance += distance.item()
            val_score += score.item()
    val_loss /= len(loader)
    val_distance /= len(loader)
    val_score /= len(loader)

    return val_loss, val_distance, val_score


def train(net, optimizer, epochs, train_loader, eval_loader, test_loader, loss_fn):
    val_loss, val_distance, val_score = evaluate(net, eval_loader, loss_fn)
    wandb.log(
        {"eval_loss": val_loss, "eval_distance": val_distance, "eval_score": val_score}
    )
    for e in range(epochs):
        s = time.time()
        global_step = e * len(train_loader.dataset)  # Num training examples
        net.train()
        for X, y in tqdm(train_loader, desc=f"Training epoch {e}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = net(X)  # B, 2
            out_lat = out[:, 0] * 90  # Scale latitude to [-90, 90]
            out_lon = out[:, 1] * 180  # Scale longitude to [-180, 180]
            out_scaled = torch.stack([out_lat, out_lon], dim=1)
            loss = loss_fn(out_scaled, y)
            loss.backward()
            optimizer.step()
            global_step += X.size(0)

            distance = get_distance(out_scaled, y)
            score = get_geoguessr_score(out_scaled, y)

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_distance": distance.item(),
                    "train_score": score.item(),
                    "step": global_step,
                }
            )
        val_loss, val_distance, val_score = evaluate(net, eval_loader, loss_fn)
        took = round(time.time() - s, 3)
        print(
            f"Epoch {e} finished, val loss: {val_loss:.3f}, val distance: {val_distance:.3f}, val score: {val_score:.3f} took {took:.3f}s"
        )
        wandb.log(
            {
                "eval_loss": val_loss,
                "eval_distance": val_distance,
                "eval_score": val_score,
                "epoch": e,
            }
        )

    return evaluate(net, test_loader, loss_fn)


def wandb_train():
    args = get_args()
    wandb.init()
    config = wandb.config

    net_name = config.net_name
    lr = config.learning_rate
    wd = config.weight_decay
    bs = config.batch_size
    dropout = config.dropout
    epochs = config.epochs
    optim = config.optimizer

    print(f"Training {net_name} on {device}, optimizer={optim}, lr={lr}, weight_decay={wd}, bs={bs}, dropout={dropout}")

    net = get_net(net_name, dropout, device)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optim == "adamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise Exception("Invalid optimizer")

    if args.mode == "geoGuessr":
        train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
            bs, net_name, directory="createDataset/geoGuessrDataset/"
        )
    elif args.mode == "googleMaps":
        train_loader, eval_loader, test_loader = get_loaders_googleMaps(
            bs, net_name, directory="createDataset/mapsDataset/"
        )

    test_loss, test_distance, test_score = train(
        net, optimizer, epochs, train_loader, eval_loader, test_loader, loss_fn
    )
    wandb.log(
        {
            "test_loss": test_loss,
            "test_distance": test_distance,
            "test_score": test_score,
        }
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["geoGuessr", "googleMaps"],
        help="Select either geoguessr or googleMaps",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "net_name": {
                "values": [
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "resnet101",
                    #"resnet152",
                    "vit_b_16",
                    "vit_b_32",
                    #"vit_l_16",
                    "efficientnet_b0",
                    "efficientnet_b1",
                    #"efficientnet_b2",
                ]
            },
            "optimizer": {
                "values": [
                    "adam",
                    "adamW",
                    "sgd",
                ]
            },
            "dropout": {"values": [0.3, 0.4, 0.5, 0.6]},
            "epochs": {"value": 4},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 5e-2,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-2,
            },
            "batch_size": {"value": 64},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="GeoGuessr")
    wandb.agent(sweep_id, wandb_train, count=5)
