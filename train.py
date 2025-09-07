import copy
import time
import torch
import wandb
import argparse

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from models import get_net
from dataset import get_loaders_geoGuessr
from utils import TrainConfig, load_config, get_optimizer

EARTH_RADIUS = 6371000
MAX_DISTANCE = 20000000.0

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def loss_fn(pred, target):
    criterion = nn.CrossEntropyLoss()

    lon_logits = pred[:, :, 0]
    lat_logits = pred[:, :, 1]

    # targets: B x 2
    lon_target = target[:, 0]
    lat_target = target[:, 1]

    loss = criterion(lon_logits, lon_target) + criterion(lat_logits, lat_target)

    return loss


# Gets average haversine distance and geoguessr score
def get_distance_and_geoguessr_score(pred_logits, truth):
    pred_classes = pred_logits.argmax(dim=1)  # B x 2
    lon_class = pred_classes[:, 0]
    lat_class = pred_classes[:, 1]

    # Convert class indices to degrees
    pred_lon = lon_class.float() / (train_config.num_classes - 1) * 360 - 180  # [-180, 180]
    pred_lat = lat_class.float() / (train_config.num_classes - 1) * 180 - 90    # [-90, 90]

    phi_pred = torch.deg2rad(pred_lat)
    phi_truth = torch.deg2rad(truth[:, 1])
    delta_phi = torch.deg2rad(truth[:, 1] - pred_lat)
    delta_lambda = torch.deg2rad(truth[:, 0] - pred_lon)

    a = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi_pred) * torch.cos(phi_truth) * torch.sin(delta_lambda / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = EARTH_RADIUS * c  # km

    scaling_factor = 2000  # km
    score = 5000 * torch.exp(-distance / scaling_factor)

    return torch.mean(distance), torch.mean(score)


def get_accs(pred_logits, targets):
    lon_logits = pred_logits[:, :, 0]  # B x NUM_CLASSES
    lat_logits = pred_logits[:, :, 1]  # B x NUM_CLASSES

    # Top-1 accuracy
    lon_top1 = lon_logits.argmax(dim=1) == targets[:, 0]
    lat_top1 = lat_logits.argmax(dim=1) == targets[:, 1]

    # Top-3 accuracy
    lon_top3 = torch.topk(lon_logits, k=3, dim=1).indices
    lat_top3 = torch.topk(lat_logits, k=3, dim=1).indices

    lon_top3_acc = (lon_top3 == targets[:, 0].unsqueeze(1)).any(dim=1)
    lat_top3_acc = (lat_top3 == targets[:, 1].unsqueeze(1)).any(dim=1)

    return (
        lon_top1.float().mean(),
        lat_top1.float().mean(),
        lon_top3_acc.float().mean(),
        lat_top3_acc.float().mean()
    )


def evaluate(net, loader):
    net.eval()
    val_loss = 0.0
    val_distance = 0.0
    val_score = 0.0
    val_lon_acc = 0.0
    val_lat_acc = 0.0
    val_lon_top3_acc = 0.0
    val_lat_top3_acc = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = net(X)  # BxCx2

            loss = loss_fn(out, y)
            distance, score = get_distance_and_geoguessr_score(out, y)
            lon_acc, lat_acc, lon_top3_acc, lat_top3_acc = get_accs(out, y)

            val_loss += loss.item()
            val_distance += distance.item()
            val_score += score.item()
            val_lon_acc += lon_acc.item()
            val_lat_acc += lat_acc.item()
            val_lon_top3_acc += lon_top3_acc.item()
            val_lat_top3_acc += lat_top3_acc.item()
    val_loss /= len(loader)
    val_distance /= len(loader)
    val_score /= len(loader)
    val_lon_acc /= len(loader)
    val_lat_acc /= len(loader)
    val_lon_top3_acc /= len(loader)
    val_lat_top3_acc /= len(loader)

    return val_loss, val_distance, val_score, val_lon_acc, val_lat_acc, val_lon_top3_acc, val_lat_top3_acc


def train(
    config: TrainConfig,
    net,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader
):
    best_loss = float('inf')
    best_net = None
    early_stop_counter = 0

    global_step = 0
    for e in range(config.epochs):
        # Evaluate
        start = time.perf_counter()
        net.eval()
        val_loss, val_distance, val_score, val_lon_acc, val_lat_acc, val_lon_top3_acc, val_lat_top3_acc = evaluate(net, eval_loader)
        taken = time.perf_counter() - start
        wandb.log(
            {
                "epoch": e,
                "steps": global_step,
                "eval_loss": val_loss,
                "eval_distance": val_distance,
                "eval_score": val_score,
                "eval_lon_acc": val_lon_acc,
                "eval_lat_acc": val_lat_acc,
                "eval_lon_top3_acc": val_lon_top3_acc,
                "eval_lat_top3_acc": val_lat_top3_acc,
            }
        )
        print(
            f"[Eval] Epoch {e},",
            f"Avg Loss: {val_loss:.3f}, Avg score: {val_score:,.2f}, Avg distance: {val_distance:,.2f}",
            f"Avg lon acc: {val_lon_acc:.2f}, Avg lon top-3 acc: {val_lon_top3_acc:.2f}, Avg lat acc: {val_lat_acc:.2f} Avg lat top-3 acc: {val_lat_top3_acc:.2f},",
            f"Time Taken: {taken:.2f}s",
        )
        start = time.perf_counter()

        # Start training
        running_loss = 0.
        running_distance = 0.
        running_score = 0.
        running_lon_acc = 0.
        running_lat_acc = 0.
        running_lon_top3_acc = 0.
        running_lat_top3_acc = 0.
        net.train()
        s = time.perf_counter()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            out = net(X)  # BxCx2

            loss = loss_fn(out, y)
            loss.backward()

            with torch.no_grad():
                distance, score = get_distance_and_geoguessr_score(out, y)
                lon_acc, lat_acc, lon_top3_acc, lat_top3_acc = get_accs(out, y)

                running_loss += loss.item()
                running_distance += distance.item()
                running_score += score.item()
                running_lon_acc += lon_acc.item()
                running_lat_acc += lat_acc.item()
                running_lon_top3_acc += lon_top3_acc.item()
                running_lat_top3_acc += lat_top3_acc.item()

            if config.gradient_clipping_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config.gradient_clipping_norm
                )

            optimizer.step()
            optimizer.zero_grad()
            global_step += bs

            if (i+1) % config.log_interval == 0:
                taken = time.perf_counter() - start
                avg_loss = running_loss / config.log_interval
                avg_distance = running_distance / config.log_interval
                avg_score = running_score / config.log_interval
                avg_lon_acc = running_lon_acc / config.log_interval
                avg_lat_acc = running_lat_acc / config.log_interval
                avg_lon_top3_acc = running_lon_top3_acc / config.log_interval
                avg_lat_top3_acc = running_lat_top3_acc / config.log_interval
                ips = config.log_interval / taken

                wandb.log(
                    {
                        "epoch": e,
                        "batch": i,
                        "train_examples": global_step,
                        "train_loss": avg_loss,
                        "train_score": avg_score,
                        "train_distance": avg_distance,
                    }
                )
                print(
                    f"Epoch {e}, step {i} (global step {global_step}),",
                    f"Avg Loss: {avg_loss:.3f}, Avg score: {avg_score:,.2f}, Avg distance: {avg_distance:,.2f}",
                    f"Avg lon acc: {avg_lon_acc:.2f}, Avg lon top-3 acc: {avg_lon_top3_acc:.2f}, Avg lat acc: {avg_lat_acc:.2f} Avg lat top-3 acc: {avg_lat_top3_acc:.2f},",
                    f"Time Taken: {taken:.2f}s, ({ips:.2f} i/s)",
                )

                running_loss = 0.
                running_distance = 0.
                running_score = 0.
                running_lon_acc = 0.
                running_lat_acc = 0.
                running_lon_top3_acc = 0.
                running_lat_top3_acc = 0.
                start = time.perf_counter()

        # Check for early stop
        if val_loss < best_loss:
            best_loss = val_loss
            best_net = copy.deepcopy(net)
        else:
            early_stop_counter += 1
            if early_stop_counter > config.early_stop:
                net = best_net
                break

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
    config_dict = {**vars(train_config)}

    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Setting up model")
    net = get_net(num_classes=train_config.num_classes, net_name=train_config.net_name, device=device)
    optimizer = get_optimizer(train_config, net)

    if args.compile:
        print("Compiling network")
        net = torch.compile(net)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params
    wandb.init(project="GeoGuessrCoordinates", name=train_config.run_name, config=config_dict)

    train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
        train_config.batch_size, train_config.num_classes, directory=train_config.dataset_dir
    )

    print("Training on device:", device)
    test_loss, test_distance, test_score, test_lon_acc, test_lat_acc, test_lon_top3_acc, test_lat_top3_acc = train(
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
            "test_lon_acc": test_lon_acc,
            "test_lat_acc": test_lat_acc,
            "test_lon_top3_acc": test_lon_top3_acc,
            "test_lat_top3_acc": test_lat_top3_acc,
        }
    )
