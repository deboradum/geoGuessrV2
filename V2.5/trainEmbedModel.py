import copy
import time
import torch
import wandb
import argparse
import numpy as np

from EmbedModel import get_embed_net
from dataset import get_loaders_geoGuessr
from config import load_config, TrainConfig
from utils import get_optimizer


def l2_distance(tensor_1, tensor_2):
    return (tensor_1 - tensor_2).pow(2).sum(1).sqrt()


def loss_fn(anchor_emb, positive_emb, negative_emb):
    criterion = torch.nn.TripletMarginLoss(margin=1., p=2., reduction="none")
    loss = criterion(anchor_emb, positive_emb, negative_emb)

    ap_l2_distance = l2_distance(anchor_emb, positive_emb)
    an_l2_distance = l2_distance(anchor_emb, negative_emb)

    return loss, ap_l2_distance, an_l2_distance


# ~TODO: Add tqdm?
def evaluate(net, loader):
    val_loss_sum = 0.0
    total_samples = 0

    all_losses = []
    all_ap_geo_dists = []
    all_an_geo_dists = []
    all_ap_l2_dists = []
    all_an_l2_dists = []

    net.eval()
    with torch.no_grad():
        for anchor_img, positive_img, negative_img, ap_geo_dist, an_geo_dist in loader:
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            bs = anchor_img.size(0)

            anchor_emb = net(anchor_img)
            positive_emb = net(positive_img)
            negative_emb = net(negative_img)

            loss, ap_l2_dist, an_l2_dist = loss_fn(anchor_emb, positive_emb, negative_emb)

            val_loss_sum += loss.sum().item()
            total_samples += bs

            all_losses.extend(loss.detach().cpu().numpy())
            all_ap_l2_dists.extend(ap_l2_dist.detach().cpu().numpy())
            all_an_l2_dists.extend(an_l2_dist.detach().cpu().numpy())

            all_ap_geo_dists.extend(ap_geo_dist.cpu().numpy())
            all_an_geo_dists.extend(an_geo_dist.cpu().numpy())

    final_metrics = {}
    final_metrics["loss_avg"] = val_loss_sum / total_samples
    final_metrics["loss_std"] = np.std(all_losses)

    final_metrics["loss_hist"] = wandb.Histogram(all_losses)
    final_metrics["geo_dist/anchor_positive"] = wandb.Histogram(all_ap_geo_dists)
    final_metrics["geo_dist/anchor_negative"] = wandb.Histogram(all_an_geo_dists)
    final_metrics["l2_distance/anchor_positive"] = wandb.Histogram(all_ap_l2_dists)
    final_metrics["l2_distance/anchor_negative"] = wandb.Histogram(all_an_l2_dists)

    # loss vs. AN geo dist
    data_an = [[loss, dist] for loss, dist in zip(all_losses, all_an_geo_dists)]
    table_an = wandb.Table(data=data_an, columns=["loss", "geo_dist_an"])
    final_metrics["scatter/loss_vs_geo_dist_an"] = wandb.plot.scatter(
        table_an, "geo_dist_an", "loss",
        title="Eval Loss vs. Geo (Anchor-Negative) Distance"
    )

    # loss vs. AP geo dist
    data_ap = [[loss, dist] for loss, dist in zip(all_losses, all_ap_geo_dists)]
    table_ap = wandb.Table(data=data_ap, columns=["loss", "geo_dist_ap"])
    final_metrics["scatter/loss_vs_geo_dist_ap"] = wandb.plot.scatter(
        table_ap, "geo_dist_ap", "loss",
        title="Eval Loss vs. Geo (Anchor-Positive) Distance"
    )

    return final_metrics


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

    # Evaluate
    start = time.perf_counter()
    net.eval()
    val_metrics = evaluate(net, eval_loader)
    net.train()
    taken = time.perf_counter() - start
    log_payload = {
        "epoch": 0,
        "train/examples": global_step,
        "eval/time": taken
    }
    for key, value in val_metrics.items():
        log_payload[f"eval/{key}"] = value
    wandb.log(log_payload)

    start = time.perf_counter()
    # ~TODO: Add tqdm?
    for e in range(config.epochs):
        total_grad_norm_before = 0.
        total_grad_norm_after = 0.

        train_loss_sum = 0.0
        total_samples = 0
        all_losses = []
        all_ap_geo_dists = []
        all_an_geo_dists = []
        all_ap_l2_dists = []
        all_an_l2_dists = []

        net.train()
        for i, (anchor_img, positive_img, negative_img, ap_geo_dist, an_geo_dist) in enumerate(train_loader):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            bs = anchor_img.size(0)

            anchor_emb = net(anchor_img)
            positive_emb = net(positive_img)
            negative_emb = net(negative_img)

            loss, ap_l2_dist, an_l2_dist = loss_fn(anchor_emb, positive_emb, negative_emb)
            batch_loss = loss.mean()
            train_loss_sum += loss.sum().item()
            total_samples += bs

            all_losses.extend(loss.detach().cpu().numpy())
            all_ap_l2_dists.extend(ap_l2_dist.detach().cpu().numpy())
            all_an_l2_dists.extend(an_l2_dist.detach().cpu().numpy())

            all_ap_geo_dists.extend(ap_geo_dist.cpu().numpy())
            all_an_geo_dists.extend(an_geo_dist.cpu().numpy())

            batch_loss.backward()

            if config.gradient_clipping_norm != 0.0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config.gradient_clipping_norm
                )
                grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None))
                total_grad_norm_before += grad_norm_before.item()
                total_grad_norm_after += grad_norm_after.item()

            optimizer.step()
            optimizer.zero_grad()
            global_step += bs

            if (i+1) % config.log_interval == 0:
                taken = time.perf_counter() - start
                ips = config.log_interval / taken

                train_metrics = {}
                train_metrics["loss_avg"] = train_loss_sum / total_samples
                train_metrics["loss_std"] = np.std(all_losses)

                train_metrics["loss_hist"] = wandb.Histogram(all_losses)
                train_metrics["geo_dist/anchor_positive"] = wandb.Histogram(all_ap_geo_dists)
                train_metrics["geo_dist/anchor_negative"] = wandb.Histogram(all_an_geo_dists)
                train_metrics["l2_distance/anchor_positive"] = wandb.Histogram(all_ap_l2_dists)
                train_metrics["l2_distance/anchor_negative"] = wandb.Histogram(all_an_l2_dists)

                # loss vs. AN geo dist
                data_an = [[loss, dist] for loss, dist in zip(all_losses, all_an_geo_dists)]
                table_an = wandb.Table(data=data_an, columns=["loss", "geo_dist_an"])
                train_metrics["scatter/loss_vs_geo_dist_an"] = wandb.plot.scatter(
                    table_an, "geo_dist_an", "loss",
                    title="Eval Loss vs. Geo (Anchor-Negative) Distance"
                )

                # loss vs. AP geo dist
                data_ap = [[loss, dist] for loss, dist in zip(all_losses, all_ap_geo_dists)]
                table_ap = wandb.Table(data=data_ap, columns=["loss", "geo_dist_ap"])
                train_metrics["scatter/loss_vs_geo_dist_ap"] = wandb.plot.scatter(
                    table_ap, "geo_dist_ap", "loss",
                    title="Eval Loss vs. Geo (Anchor-Positive) Distance"
                )

                log_payload = {
                    "epoch": e,
                    "batch": i,
                    "train/iterations_per_second": ips,
                    "train/examples": global_step,
                    "train/grad_norm_before_clip": total_grad_norm_before / config.log_interval,
                    "train/grad_norm_after_clip": total_grad_norm_after / config.log_interval,
                }
                for key, value in train_metrics.items():
                    log_payload[f"train/{key}"] = value
                wandb.log(log_payload)

                total_grad_norm_before = 0.
                total_grad_norm_after = 0.

                train_loss_sum = 0.0
                total_samples = 0
                all_losses = []
                all_ap_geo_dists = []
                all_an_geo_dists = []
                all_ap_l2_dists = []
                all_an_l2_dists = []
                start = time.perf_counter()

        # Evaluate
        start = time.perf_counter()
        net.eval()
        val_metrics = evaluate(net, eval_loader)
        net.train()
        taken = time.perf_counter() - start
        log_payload = {
            "epoch": e,
            "train/examples": global_step,
            "eval/time": taken
        }
        for key, value in val_metrics.items():
            log_payload[f"eval/{key}"] = value
        wandb.log(log_payload)

        # Check for early stop
        if val_metrics["loss_avg"] < best_loss:
            best_loss = val_metrics["loss_avg"]
            best_net = copy.deepcopy(net)
        else:
            early_stop_counter += 1
            if early_stop_counter > config.early_stop:
                net = best_net
                break

        start = time.perf_counter()

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
    config = load_config(args.config)
    config_dict = {**vars(config)}

    torch.manual_seed(config.trainConfig.seed)
    torch.backends.cudnn.deterministic = True
    # TODO: set other seeds

    device = config.device

    print("Setting up model")
    net = get_embed_net(
        embedding_dim=config.embedModelConfig.embedding_dim,
        freeze_weights=config.embedModelConfig.freeze_weights,
        size=config.embedModelConfig.size,
        device=device,
    )
    optimizer = get_optimizer(config.trainConfig, net)

    if args.compile:
        print("Compiling network")
        net = torch.compile(net)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params
    wandb.init(project="GeoGuessrCoordinatesV2.5", name=config.trainConfig.run_name, config=config_dict)
    wandb.watch(net, log="all", log_freq=config.trainConfig.log_interval * 10) # Log grads & params

    train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
        config.trainConfig.batch_size, directory=config.trainConfig.dataset_dir
    )

    print("Training on device:", device)
    test_metrics = train(
        config=config.trainConfig,
        net=net,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader
    )
    wandb.log({f"test/{key}": value for key, value in test_metrics.items()})
