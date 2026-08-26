import copy
import time
import torch
import wandb
import argparse

import torch.nn as nn
from typing import DefaultDict
import torch.nn.functional as F
from collections import defaultdict

from models import get_net
from dataset import get_loaders
from utils import TrainConfig, load_config, get_optimizer, gcs_to_cartesian_tensor, cartesian_to_gcs_tensor, save_predictions

EARTH_RADIUS = 6371000  # meters

s2_criterion = torch.nn.CrossEntropyLoss()

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Haversine distance loss and geoguessr score
def loss_fn(pred, target):
    pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
    pred_lat_deg, pred_lon_deg = cartesian_to_gcs_tensor(pred_x, pred_y, pred_z)

    true_lon_deg, true_lat_deg = target[:, 0], target[:, 1]
    true_x, true_y, true_z = gcs_to_cartesian_tensor(true_lat_deg, true_lon_deg)
    target_cartesian = torch.stack([true_x, true_y, true_z], dim=1)
    target_cartesian = F.normalize(target_cartesian, p=2, dim=1, eps=1e-8)

    mse_loss = torch.nn.MSELoss()(pred, target_cartesian)

    pred_normalized = F.normalize(pred, p=2, dim=1, eps=1e-8)

    chordal_dist = torch.norm(pred_normalized - target_cartesian, p=2, dim=1)
    clamped_ratio = torch.clamp(chordal_dist / 2.0, min=0.0, max=1.0 - 1e-5)

    c = 2.0 * torch.asin(clamped_ratio)

    with torch.no_grad():
        distance = EARTH_RADIUS * c / 1000.0  # km
        scaling_factor = 2000.0  # km
        score = 5000.0 * torch.exp(-distance / scaling_factor)

    return {
        "mse_loss": mse_loss,
        "distance_rad_avg": c.mean(),
        "distance_rad_std": c.std(),
        "distance_avg": distance.mean(),
        "distance_std": distance.std(),
        "distance_median": distance.median(),
        "distance_p10": torch.quantile(distance, 0.10),
        "distance_p20": torch.quantile(distance, 0.20),
        "distance_p80": torch.quantile(distance, 0.80),
        "distance_p90": torch.quantile(distance, 0.90),
        "score_avg": score.mean(),
        "score_std": score.std(),
        "score_median": score.median(),
        "score_p10": torch.quantile(score, 0.10),
        "score_p20": torch.quantile(score, 0.20),
        "score_p80": torch.quantile(score, 0.80),
        "score_p90": torch.quantile(score, 0.90),
        "abs_err_lon_deg": torch.abs(target[:, 0] - pred[:, 0]).mean(),
        "abs_err_lat_deg": torch.abs(target[:, 1] - pred[:, 1]).mean(),
        "pred_lon_std": pred_lon_deg.std(),
        "true_lon_std": true_lon_deg.std(),
        "pred_lat_std": pred_lat_deg.std(),
        "true_lat_std": true_lat_deg.std(),
        "distances_raw": distance,
    }


def evaluate(net, loader, dist_loss_weight, s2_loss_weight, load_balance_loss_weight, epoch: int|str, run_name: str, num_viz_batches: int):
    val_metrics_sums = defaultdict(float)
    total_samples = 0
    all_distances_tensors = []

    net.eval()
    with torch.no_grad():
        for i, (X, (y_coords, y_s2)) in enumerate(loader):
            X, y_coords, y_s2 = X.to(device), y_coords.to(device), y_s2.to(device)
            bs = X.size(0)

            out, s2_logits, load_metrics = net(X)
            batch_metrics = loss_fn(out, y_coords)
            dist_loss = batch_metrics["distance_rad_avg"]

            s2_loss = s2_criterion(s2_logits, y_s2)
            batch_metrics["s2_loss"] = s2_loss

            pred_labels = torch.argmax(s2_logits, dim=1)
            correct = (pred_labels == y_s2).sum()
            batch_metrics["s2_accuracy"] = correct.float() / bs

            aux_loss = load_metrics["load_balancing_loss"]

            gamma = dist_loss_weight
            alpha = s2_loss_weight
            beta = load_balance_loss_weight

            total_loss = (gamma * dist_loss) + (alpha * s2_loss) + (beta * aux_loss)

            batch_metrics["total_loss"] = total_loss

            for key, value in load_metrics.items():
                val_metrics_sums[key] += value.item() * bs

            all_distances_tensors.append(batch_metrics["distances_raw"].detach().cpu())

            for key, value_tensor in batch_metrics.items():
                if key != "distances_raw":
                    val_metrics_sums[key] += value_tensor.item() * bs

            total_samples += bs

            if i < num_viz_batches:
                save_predictions(X, out, y_coords, distances=batch_metrics["distances_raw"], output_dir=f"visualizations_{run_name}/{epoch}/")

    final_metrics_avg = {}
    if total_samples > 0:
        for key, total_sum in val_metrics_sums.items():
            final_metrics_avg[key] = total_sum / total_samples

    if all_distances_tensors:
        all_distances = torch.cat(all_distances_tensors).tolist()
    else:
        all_distances = []

    return final_metrics_avg, all_distances


def train(
    config: TrainConfig,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    viz_batches: int = 1
):
    best_distance = float('inf')
    best_state_dict = copy.deepcopy(net.state_dict())
    early_stop_counter = 0
    global_step = 0

    # Evaluate
    start = time.perf_counter()
    net.eval()
    val_metrics, all_eval_distances = evaluate(net, eval_loader, config.dist_loss_weight, config.s2_loss_weight, config.load_balance_loss_weight, epoch="initial", run_name=config.run_name, num_viz_batches=viz_batches)
    net.train()
    taken = time.perf_counter() - start
    wandb.log(
        {
            "epoch": 0,
            "examples": global_step,
            "eval/mse_loss": val_metrics.get("mse_loss", -1),
            "eval/total_loss": val_metrics.get("total_loss", -1),
            "eval/s2_loss": val_metrics.get("s2_loss", -1),
            "eval/s2_accuracy": val_metrics.get("s2_accuracy", -1),
            "eval/load_balancing_loss": val_metrics.get("load_balancing_loss", -1),
            "eval/expert_load_cv": val_metrics.get("expert_load_cv", -1),
            "eval/dead_experts": val_metrics.get("dead_experts", -1),
            "eval/router_prob_entropy": val_metrics.get("router_prob_entropy", -1),
            "eval/distance_rad_avg": val_metrics.get("distance_rad_avg", -1),
            "eval/distance_rad_std": val_metrics.get("distance_rad_std", -1),
            "eval/distance_avg": val_metrics.get("distance_avg", -1),
            "eval/distance_std": val_metrics.get("distance_std", -1),
            "eval/distance_median": val_metrics.get("distance_median", -1),
            "eval/distance_p10": val_metrics.get("distance_p10", -1),
            "eval/distance_p20": val_metrics.get("distance_p20", -1),
            "eval/distance_p80": val_metrics.get("distance_p80", -1),
            "eval/distance_p90": val_metrics.get("distance_p90", -1),
            "eval/score_avg": val_metrics.get("score_avg", -1),
            "eval/score_std": val_metrics.get("score_std", -1),
            "eval/score_median": val_metrics.get("score_median", -1),
            "eval/score_p10": val_metrics.get("score_p10", -1),
            "eval/score_p20": val_metrics.get("score_p20", -1),
            "eval/score_p80": val_metrics.get("score_p80", -1),
            "eval/score_p90": val_metrics.get("score_p90", -1),
            "eval/abs_err_lon_deg": val_metrics.get("abs_err_lon_deg", -1),
            "eval/abs_err_lat_deg": val_metrics.get("abs_err_lat_deg", -1),
            "eval/pred_lon_std": val_metrics.get("pred_lon_std", -1),
            "eval/true_lon_std": val_metrics.get("true_lon_std", -1),
            "eval/pred_lat_std": val_metrics.get("pred_lat_std", -1),
            "eval/true_lat_std": val_metrics.get("true_lat_std", -1),
            "eval/distance_histogram": wandb.Histogram(all_eval_distances),
        }
    )
    print(
        f"[Eval] Epoch 0, Time: {taken:.2f}s\n"
        f"  dist loss:   {val_metrics.get('mse_loss', 0.0):.2f}\n"
        f"  aux loss:    {val_metrics.get('load_balancing_loss', 0.0):.2f}\n"
        f"  S2 loss:     {val_metrics.get('s2_loss', 0.0):.2f}\n"
        f"  S2 acc:      {val_metrics.get('s2_accuracy', 0.0):.3f}\n"
        f"  total loss:  {val_metrics.get('total_loss', 0.0):.2f}\n"
        f"  Score:       {val_metrics.get('score_avg', 0.0):,.2f} ± {val_metrics.get('score_std', 0.0):,.2f}\n"
        f"  Distance:    {val_metrics.get('distance_avg', 0.0):,.2f} ± {val_metrics.get('distance_std', 0.0):,.2f} km\n"
        f"  Score (p20/p50/p80):    {val_metrics.get('score_p20', 0.0):,.2f} / {val_metrics.get('score_median', 0.0):,.2f} / {val_metrics.get('score_p80', 0.0):,.2f}\n"
        f"  Distance (p20/p50/p80): {val_metrics.get('distance_p20', 0.0):,.2f} / {val_metrics.get('distance_median', 0.0):,.2f} / {val_metrics.get('distance_p80', 0.0):,.2f} km\n"
    )

    start = time.perf_counter()
    for e in range(config.epochs):
        running_metrics_sums: DefaultDict[str, float] = defaultdict(float)
        total_grad_norm_before = 0.
        total_grad_norm_after = 0.

        net.train()
        for i, (X, (y_coords, y_s2)) in enumerate(train_loader):
            X, y_coords, y_s2 = X.to(device), y_coords.to(device), y_s2.to(device)
            bs = X.shape[0]

            optimizer.zero_grad()
            out, s2_logits, load_metrics = net(X)  # Bx3
            for key, value in load_metrics.items():
                running_metrics_sums[key] += value.item()

            batch_metrics = loss_fn(out, y_coords)
            dist_loss = batch_metrics["distance_rad_avg"]

            s2_loss = s2_criterion(s2_logits, y_s2)
            batch_metrics["s2_loss"] = s2_loss

            pred_labels = torch.argmax(s2_logits, dim=1)
            correct = (pred_labels == y_s2).sum()
            batch_metrics["s2_accuracy"] = correct.float() / bs

            aux_loss = load_metrics["load_balancing_loss"]

            gamma = config.dist_loss_weight
            alpha = config.s2_loss_weight
            beta = config.load_balance_loss_weight

            scaled_dist_loss = gamma * dist_loss
            scaled_s2_loss = alpha * s2_loss
            scaled_aux_loss = beta * aux_loss
            # Optimize against distance, s2 cross entropy and load balancing loss.
            total_loss = scaled_dist_loss + scaled_s2_loss + scaled_aux_loss
            total_loss.backward()

            running_metrics_sums["total_loss"] += total_loss.item()

            for key, value in batch_metrics.items():
                if key != "distances_raw":
                    running_metrics_sums[key] += value.item()

            if config.gradient_clipping_norm != 0.0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config.gradient_clipping_norm
                )
                grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)) # type: ignore
                total_grad_norm_before += grad_norm_before.item()
                total_grad_norm_after += grad_norm_after.item()

            optimizer.step()
            global_step += bs

            if (i+1) % config.log_interval == 0:
                taken = time.perf_counter() - start
                ips = config.log_interval / taken

                train_metrics = {}
                for key, sum_of_avgs in running_metrics_sums.items():
                    train_metrics[key] = sum_of_avgs / config.log_interval

                wandb.log(
                    {
                        "epoch": e,
                        "batch": i,
                        "train/iterations_per_second": ips,
                        "examples": global_step,
                        "train/total_loss": train_metrics.get("total_loss", -1),
                        "train/mse_loss": train_metrics.get("mse_loss", -1),
                        "train/load_balancing_loss": train_metrics.get("load_balancing_loss", -1),
                        "train/scaled_load_balancing_loss": scaled_aux_loss,
                        "train/s2_loss": train_metrics.get("s2_loss", -1),
                        "train/scaled_s2_loss": scaled_s2_loss,
                        "train/s2_accuracy": train_metrics.get("s2_accuracy", -1),
                        "train/expert_load_cv": train_metrics.get("expert_load_cv", -1),
                        "train/dead_experts": train_metrics.get("dead_experts", -1),
                        "train/router_prob_entropy": train_metrics.get("router_prob_entropy", -1),
                        "train/distance_rad_avg": train_metrics.get("distance_rad_avg", -1),
                        "train/distance_rad_std": train_metrics.get("distance_rad_std", -1),
                        "train/distance_avg": train_metrics.get("distance_avg", -1),
                        "train/distance_std": train_metrics.get("distance_std", -1),
                        "train/distance_median": train_metrics.get("distance_median", -1),
                        "train/distance_p10": train_metrics.get("distance_p10", -1),
                        "train/distance_p20": train_metrics.get("distance_p20", -1),
                        "train/distance_p80": train_metrics.get("distance_p80", -1),
                        "train/distance_p90": train_metrics.get("distance_p90", -1),
                        "train/score_avg": train_metrics.get("score_avg", -1),
                        "train/score_std": train_metrics.get("score_std", -1),
                        "train/score_median": train_metrics.get("score_median", -1),
                        "train/score_p10": train_metrics.get("score_p10", -1),
                        "train/score_p20": train_metrics.get("score_p20", -1),
                        "train/score_p80": train_metrics.get("score_p80", -1),
                        "train/score_p90": train_metrics.get("score_p90", -1),
                        "train/abs_err_lon_deg": train_metrics.get("abs_err_lon_deg", -1),
                        "train/abs_err_lat_deg": train_metrics.get("abs_err_lat_deg", -1),
                        "train/pred_lon_std": train_metrics.get("pred_lon_std", -1),
                        "train/true_lon_std": train_metrics.get("true_lon_std", -1),
                        "train/pred_lat_std": train_metrics.get("pred_lat_std", -1),
                        "train/true_lat_std": train_metrics.get("true_lat_std", -1),
                        "optimizer/grad_norm_before_clip": total_grad_norm_before / config.log_interval,
                        "optimizer/grad_norm_after_clip": total_grad_norm_after / config.log_interval,
                    }
                )
                print(
                    f"Epoch {e}, step {i} (Global {global_step}), Time: {taken:.2f}s ({ips:.2f} i/s)\n"
                    f"  dist loss:   {train_metrics.get('distance_rad_avg', 0.0):.2f} ({train_metrics.get('distance_rad_avg', 0.0) * config.dist_loss_weight:.2f})\n"
                    f"  aux loss:    {train_metrics.get('load_balancing_loss', 0.0):.2f} ({train_metrics.get('load_balancing_loss', 0.0) * config.load_balance_loss_weight:.2f})\n"
                    f"  S2 loss:     {train_metrics.get('s2_loss', 0.0):.2f} ({train_metrics.get('s2_loss', 0.0) * config.s2_loss_weight:.2f})\n"
                    f"  S2 acc:      {train_metrics.get('s2_accuracy', 0.0):.3f}\n"
                    f"  total loss:  {train_metrics.get('total_loss', 0.0):.2f}\n"
                    f"  Score:       {train_metrics.get('score_avg', 0.0):,.2f} ± {train_metrics.get('score_std', 0.0):,.2f}\n"
                    f"  Distance:    {train_metrics.get('distance_avg', 0.0):,.2f} ± {train_metrics.get('distance_std', 0.0):,.2f} km\n"
                    f"  Score (p20/p50/p80):    {train_metrics.get('score_p20', 0.0):,.2f} / {train_metrics.get('score_median', 0.0):,.2f} / {train_metrics.get('score_p80', 0.0):,.2f}\n"
                    f"  Distance (p20/p50/p80): {train_metrics.get('distance_p20', 0.0):,.2f} / {train_metrics.get('distance_median', 0.0):,.2f} / {train_metrics.get('distance_p80', 0.0):,.2f} km\n"
                )

                running_metrics_sums = defaultdict(float)
                total_grad_norm_before = 0.
                total_grad_norm_after = 0.
                start = time.perf_counter()

        # Evaluate
        start = time.perf_counter()
        net.eval()

        val_metrics, all_eval_distances = evaluate(net, eval_loader, config.dist_loss_weight, config.s2_loss_weight, config.load_balance_loss_weight, epoch=e, run_name=config.run_name, num_viz_batches=viz_batches)
        net.train()
        taken = time.perf_counter() - start
        wandb.log(
            {
                "epoch": e+1,
                "examples": global_step,
                "eval/mse_loss": val_metrics.get("mse_loss", -1),
                "eval/s2_loss": val_metrics.get("s2_loss", -1),
                "eval/total_loss": val_metrics.get("total_loss", -1),
                "eval/s2_accuracy": val_metrics.get("s2_accuracy", -1),
                "eval/load_balancing_loss": val_metrics.get("load_balancing_loss", -1),
                "eval/expert_load_cv": val_metrics.get("expert_load_cv", -1),
                "eval/dead_experts": val_metrics.get("dead_experts", -1),
                "eval/router_prob_entropy": val_metrics.get("router_prob_entropy", -1),
                "eval/distance_rad_avg": val_metrics.get("distance_rad_avg", -1),
                "eval/distance_rad_std": val_metrics.get("distance_rad_std", -1),
                "eval/distance_avg": val_metrics.get("distance_avg", -1),
                "eval/distance_std": val_metrics.get("distance_std", -1),
                "eval/distance_median": val_metrics.get("distance_median", -1),
                "eval/distance_p10": val_metrics.get("distance_p10", -1),
                "eval/distance_p20": val_metrics.get("distance_p20", -1),
                "eval/distance_p80": val_metrics.get("distance_p80", -1),
                "eval/distance_p90": val_metrics.get("distance_p90", -1),
                "eval/score_avg": val_metrics.get("score_avg", -1),
                "eval/score_std": val_metrics.get("score_std", -1),
                "eval/score_median": val_metrics.get("score_median", -1),
                "eval/score_p10": val_metrics.get("score_p10", -1),
                "eval/score_p20": val_metrics.get("score_p20", -1),
                "eval/score_p80": val_metrics.get("score_p80", -1),
                "eval/score_p90": val_metrics.get("score_p90", -1),
                "eval/abs_err_lon_deg": val_metrics.get("abs_err_lon_deg", -1),
                "eval/abs_err_lat_deg": val_metrics.get("abs_err_lat_deg", -1),
                "eval/pred_lon_std": val_metrics.get("pred_lon_std", -1),
                "eval/true_lon_std": val_metrics.get("true_lon_std", -1),
                "eval/pred_lat_std": val_metrics.get("pred_lat_std", -1),
                "eval/true_lat_std": val_metrics.get("true_lat_std", -1),
                "eval/distance_histogram": wandb.Histogram(all_eval_distances),
            }
        )
        print(
            f"[Eval] Epoch {e+1}, Time: {taken:.2f}s\n"
            f"  dist loss:   {val_metrics.get('mse_loss', 0.0):.2f}\n"
            f"  aux loss:    {val_metrics.get('load_balancing_loss', 0.0):.2f}\n"
            f"  S2 loss:     {val_metrics.get('s2_loss', 0.0):.2f}\n"
            f"  S2 acc:      {val_metrics.get('s2_accuracy', 0.0):.3f}\n"
            f"  total loss:  {val_metrics.get('total_loss', 0.0):.2f}\n"
            f"  Score:       {val_metrics.get('score_avg', 0.0):,.2f} ± {val_metrics.get('score_std', 0.0):,.2f}\n"
            f"  Distance:    {val_metrics.get('distance_avg', 0.0):,.2f} ± {val_metrics.get('distance_std', 0.0):,.2f} km\n"
            f"  Score (p20/p50/p80):    {val_metrics.get('score_p20', 0.0):,.2f} / {val_metrics.get('score_median', 0.0):,.2f} / {val_metrics.get('score_p80', 0.0):,.2f}\n"
            f"  Distance (p20/p50/p80): {val_metrics.get('distance_p20', 0.0):,.2f} / {val_metrics.get('distance_median', 0.0):,.2f} / {val_metrics.get('distance_p80', 0.0):,.2f} km\n"
        )
        start = time.perf_counter()

        # Check for early stop
        if val_metrics["distance_avg"] < best_distance:
            best_distance = val_metrics["distance_avg"]
            best_state_dict = copy.deepcopy(net.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > config.early_stop:
                net.load_state_dict(best_state_dict)
                break

    # Load best model before test evaluation
    net.load_state_dict(best_state_dict)

    # torch.save(best_state_dict, f"best_model_{config.run_name}.pth")

    net.eval()
    return evaluate(net, test_loader, config.dist_loss_weight, config.s2_loss_weight, config.load_balance_loss_weight, epoch="test", run_name=config.run_name, num_viz_batches=max(1, viz_batches))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to train config", required=True
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model before training"
    )
    parser.add_argument(
        "--viz_batches", type=int, default=1, help="Amount of batches to save visualizations for"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_config = load_config(args.config)
    config_dict = {**vars(train_config)}

    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, eval_loader, test_loader = get_loaders(
        directory=train_config.dataset_dir, s2_cell_level=train_config.s2_cell_level
    )

    print("Setting up model")
    net = get_net(train_loader.dataset.num_unique_s2_classes, config=train_config, device=device) # type: ignore
    optimizer = get_optimizer(train_config, net)

    if args.compile:
        print("Compiling network")
        net = torch.compile(net)

    num_params = sum(p.numel() for p in net.parameters()) # type: ignore
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params
    size = train_config.net_name.split("-")[-1]
    wandb.init(
        project="GeoGuessrCoordinatesV2",
        name=train_config.run_name,
        config=config_dict,
        tags=[size],
        settings=wandb.Settings(x_disable_stats=True),
    )
    # wandb.watch(net, log="all", log_freq=train_config.log_interval * 10) # type: ignore # Log grads & params

    print("Training on device:", device)
    test_metrics, all_test_distances = train(
        config=train_config,
        net=net, # type: ignore
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        viz_batches=args.viz_batches
    )
    wandb.log(
        {
            "test/mse_loss": test_metrics.get("mse_loss", -1),
            "test/s2_loss": test_metrics.get("s2_loss", -1),
            "test/s2_accuracy": test_metrics.get("s2_accuracy", -1),
            "test/total_loss": test_metrics.get("total_loss", -1),
            "test/load_balancing_loss": test_metrics.get("load_balancing_loss", -1),
            "test/expert_load_cv": test_metrics.get("expert_load_cv", -1),
            "test/dead_experts": test_metrics.get("dead_experts", -1),
            "test/router_prob_entropy": test_metrics.get("router_prob_entropy", -1),
            "test/distance_rad_avg": test_metrics.get("distance_rad_avg", -1),
            "test/distance_rad_std": test_metrics.get("distance_rad_std", -1),
            "test/distance_avg": test_metrics.get("distance_avg", -1),
            "test/distance_std": test_metrics.get("distance_std", -1),
            "test/distance_median": test_metrics.get("distance_median", -1),
            "test/distance_p10": test_metrics.get("distance_p10", -1),
            "test/distance_p20": test_metrics.get("distance_p20", -1),
            "test/distance_p80": test_metrics.get("distance_p80", -1),
            "test/distance_p90": test_metrics.get("distance_p90", -1),
            "test/score_avg": test_metrics.get("score_avg", -1),
            "test/score_std": test_metrics.get("score_std", -1),
            "test/score_median": test_metrics.get("score_median", -1),
            "test/score_p10": test_metrics.get("score_p10", -1),
            "test/score_p20": test_metrics.get("score_p20", -1),
            "test/score_p80": test_metrics.get("score_p80", -1),
            "test/score_p90": test_metrics.get("score_p90", -1),
            "test/abs_err_lon_deg": test_metrics.get("abs_err_lon_deg", -1),
            "test/abs_err_lat_deg": test_metrics.get("abs_err_lat_deg", -1),
            "test/pred_lon_std": test_metrics.get("pred_lon_std", -1),
            "test/true_lon_std": test_metrics.get("true_lon_std", -1),
            "test/pred_lat_std": test_metrics.get("pred_lat_std", -1),
            "test/true_lat_std": test_metrics.get("true_lat_std", -1),
            "test/distance_histogram": wandb.Histogram(all_test_distances),
        }
    )
