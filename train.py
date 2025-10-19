import copy
import time
import torch
import wandb
import argparse

from collections import defaultdict

from models import get_net
from dataset import get_loaders_geoGuessr
from utils import TrainConfig, load_config, get_optimizer

EARTH_RADIUS = 6371000  # meters

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

    with torch.no_grad():
        distance = EARTH_RADIUS * c / 1000  # km
        scaling_factor = 2000  # km
        score = 5000 * torch.exp(-distance / scaling_factor)

    return {
        "loss_avg": c.mean(),
        "loss_std": c.std(),
        "distance_avg": distance.mean(),
        "distance_std": distance.std(),
        "distance_median": distance.median(),
        "distance_p80": torch.quantile(distance, 0.80),
        "distance_p90": torch.quantile(distance, 0.90),
        "distance_p95": torch.quantile(distance, 0.95),
        "score_avg": score.mean(),
        "score_std": score.std(),
        "score_median": score.median(),
        "score_p80": torch.quantile(score, 0.80),
        "score_p90": torch.quantile(score, 0.90),
        "score_p95": torch.quantile(score, 0.95),
        "abs_err_lon_deg": torch.abs(target[:, 0] - pred[:, 0]).mean(),
        "abs_err_lat_deg": torch.abs(target[:, 1] - pred[:, 1]).mean(),
        "pred_lon_std": pred_lon.std(),
        "true_lon_std": true_lon.std(),
        "pred_lat_std": pred_lat.std(),
        "true_lat_std": true_lat.std(),
        "distances_raw": distance,
    }


def evaluate(net, loader):
    val_metrics_sums = defaultdict(float)
    total_samples = 0
    all_distances = []

    net.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = net(X)  # BxCx2
            batch_metrics = loss_fn(out, y)
            bs = X.size(0)

            all_distances.extend(batch_metrics["distances_raw"].cpu().tolist())
            for key, value_tensor in batch_metrics.items():
                if key != "distances_raw":
                    val_metrics_sums[key] += value_tensor.item() * bs

            total_samples += bs

    final_metrics_avg = {}
    if total_samples > 0:
        for key, total_sum in val_metrics_sums.items():
            final_metrics_avg[key] = total_sum / total_samples

    return final_metrics_avg, all_distances


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
    val_metrics, all_eval_distances = evaluate(net, eval_loader)
    net.train()
    taken = time.perf_counter() - start
    wandb.log(
        {
            "epoch": 0,
            "train/examples": global_step,
            "eval/loss_avg": val_metrics.get("loss_avg", -1),
            "eval/loss_std": val_metrics.get("loss_std", -1),
            "eval/distance_avg": val_metrics.get("distance_avg", -1),
            "eval/distance_std": val_metrics.get("distance_std", -1),
            "eval/distance_p80": val_metrics.get("distance_p80", -1),
            "eval/distance_p90": val_metrics.get("distance_p90", -1),
            "eval/distance_p95": val_metrics.get("distance_p95", -1),
            "eval/score_avg": val_metrics.get("score_avg", -1),
            "eval/score_std": val_metrics.get("score_std", -1),
            "eval/score_p80": val_metrics.get("score_p80", -1),
            "eval/score_p90": val_metrics.get("score_p90", -1),
            "eval/score_p95": val_metrics.get("score_p95", -1),
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
        f"  Loss:     {val_metrics.get('loss_avg', 0.0):.2f} ± {val_metrics.get('loss_std', 0.0):.2f}\n"
        f"  Score:    {val_metrics.get('score_avg', 0.0):,.2f} ± {val_metrics.get('score_std', 0.0):,.2f}\n"
        f"  Distance: {val_metrics.get('distance_avg', 0.0):,.2f} ± {val_metrics.get('distance_std', 0.0):,.2f} km\n"
        f"  Distance (p80/p90/p95): {val_metrics.get('distance_p80', 0.0):,.2f} / {val_metrics.get('distance_p90', 0.0):,.2f} / {val_metrics.get('distance_p95', 0.0):,.2f} km\n"
        f"  Std (Pred Lon/Lat): {val_metrics.get('pred_lon_std', 0.0):.2f} / {val_metrics.get('pred_lat_std', 0.0):.2f}\n"
        f"  Std (True Lon/Lat): {val_metrics.get('true_lon_std', 0.0):.2f} / {val_metrics.get('true_lat_std', 0.0):.2f}"
    )

    start = time.perf_counter()
    for e in range(config.epochs):
        running_metrics_sums = defaultdict(float)
        total_grad_norm_before = 0.
        total_grad_norm_after = 0.

        net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            out = net(X)  # Bx2
            batch_metrics = loss_fn(out, y)

            loss = batch_metrics["loss_avg"]
            loss.backward()

            for key, value in batch_metrics.items():
                if key != "distances_raw":
                    running_metrics_sums[key] += value.item()

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
                for key, sum_of_avgs in running_metrics_sums.items():
                    train_metrics[key] = sum_of_avgs / config.log_interval

                wandb.log(
                    {
                        "epoch": e,
                        "batch": i,
                        "train/iterations_per_second": ips,
                        "train/examples": global_step,
                        "train/loss_avg": train_metrics.get("loss_avg", -1),
                        "train/loss_std": train_metrics.get("loss_std", -1),
                        "train/distance_avg": train_metrics.get("distance_avg", -1),
                        "train/distance_std": train_metrics.get("distance_std", -1),
                        "train/distance_p80": train_metrics.get("distance_p80", -1),
                        "train/distance_p90": train_metrics.get("distance_p90", -1),
                        "train/distance_p95": train_metrics.get("distance_p95", -1),
                        "train/score_avg": train_metrics.get("score_avg", -1),
                        "train/score_std": train_metrics.get("score_std", -1),
                        "train/score_p80": train_metrics.get("score_p80", -1),
                        "train/score_p90": train_metrics.get("score_p90", -1),
                        "train/score_p95": train_metrics.get("score_p95", -1),
                        "train/abs_err_lon_deg": train_metrics.get("abs_err_lon_deg", -1),
                        "train/abs_err_lat_deg": train_metrics.get("abs_err_lat_deg", -1),
                        "train/pred_lon_std": train_metrics.get("pred_lon_std", -1),
                        "train/true_lon_std": train_metrics.get("true_lon_std", -1),
                        "train/pred_lat_std": train_metrics.get("pred_lat_std", -1),
                        "train/true_lat_std": train_metrics.get("true_lat_std", -1),
                        "train/grad_norm_before_clip": total_grad_norm_before / config.log_interval,
                        "train/grad_norm_after_clip": total_grad_norm_after / config.log_interval,
                    }
                )
                print(
                    f"Epoch {e}, step {i} (Global {global_step}), Time: {taken:.2f}s ({ips:.2f} i/s)\n"
                    f"  Loss:     {train_metrics.get('loss_avg', 0.0):.2f} ± {train_metrics.get('loss_std', 0.0):.2f}\n"
                    f"  Score:    {train_metrics.get('score_avg', 0.0):,.2f} ± {train_metrics.get('score_std', 0.0):,.2f}\n"
                    f"  Distance: {train_metrics.get('distance_avg', 0.0):,.2f} ± {train_metrics.get('distance_std', 0.0):,.2f} km\n"
                    f"  Distance (p80/p90/p95): {train_metrics.get('distance_p80', 0.0):,.2f} / {train_metrics.get('distance_p90', 0.0):,.2f} / {train_metrics.get('distance_p95', 0.0):,.2f} km\n"
                    f"  Std (Pred Lon/Lat): {train_metrics.get('pred_lon_std', 0.0):.2f} / {train_metrics.get('pred_lat_std', 0.0):.2f}\n"
                    f"  Std (True Lon/Lat): {train_metrics.get('true_lon_std', 0.0):.2f} / {train_metrics.get('true_lat_std', 0.0):.2f}"
                )

                running_metrics_sums = defaultdict(float)
                total_grad_norm_before = 0.
                total_grad_norm_after = 0.
                start = time.perf_counter()

        # Evaluate
        start = time.perf_counter()
        net.eval()
        val_metrics, all_eval_distances = evaluate(net, eval_loader)
        net.train()
        taken = time.perf_counter() - start
        wandb.log(
            {
                "epoch": e+1,
                "train/examples": global_step,
                "eval/loss_avg": val_metrics.get("loss_avg", -1),
                "eval/loss_std": val_metrics.get("loss_std", -1),
                "eval/distance_avg": val_metrics.get("distance_avg", -1),
                "eval/distance_std": val_metrics.get("distance_std", -1),
                "eval/distance_p80": val_metrics.get("distance_p80", -1),
                "eval/distance_p90": val_metrics.get("distance_p90", -1),
                "eval/distance_p95": val_metrics.get("distance_p95", -1),
                "eval/score_avg": val_metrics.get("score_avg", -1),
                "eval/score_std": val_metrics.get("score_std", -1),
                "eval/score_p80": val_metrics.get("score_p80", -1),
                "eval/score_p90": val_metrics.get("score_p90", -1),
                "eval/score_p95": val_metrics.get("score_p95", -1),
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
            f"  Loss:     {val_metrics.get('loss_avg', 0.0):.2f} ± {val_metrics.get('loss_std', 0.0):.2f}\n"
            f"  Score:    {val_metrics.get('score_avg', 0.0):,.2f} ± {val_metrics.get('score_std', 0.0):,.2f}\n"
            f"  Distance: {val_metrics.get('distance_avg', 0.0):,.2f} ± {val_metrics.get('distance_std', 0.0):,.2f} km\n"
            f"  Distance (p80/p90/p95): {val_metrics.get('distance_p80', 0.0):,.2f} / {val_metrics.get('distance_p90', 0.0):,.2f} / {val_metrics.get('distance_p95', 0.0):,.2f} km\n"
            f"  Std (Pred Lon/Lat): {val_metrics.get('pred_lon_std', 0.0):.2f} / {val_metrics.get('pred_lat_std', 0.0):.2f}\n"
            f"  Std (True Lon/Lat): {val_metrics.get('true_lon_std', 0.0):.2f} / {val_metrics.get('true_lat_std', 0.0):.2f}"
        )
        start = time.perf_counter()

        # Check for early stop
        if val_metrics["distance_avg"] < best_distance:
            best_distance = val_metrics["distance_avg"]
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
    config_dict = {**vars(train_config)}

    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Setting up model")
    net = get_net(freeze_weights=train_config.freeze_weights, net_name=train_config.net_name, device=device)
    optimizer = get_optimizer(train_config, net)

    if args.compile:
        print("Compiling network")
        net = torch.compile(net)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params
    wandb.init(project="GeoGuessrCoordinates", name=train_config.run_name, config=config_dict)
    wandb.watch(net, log="all", log_freq=train_config.log_interval * 10) # Log grads & params

    train_loader, eval_loader, test_loader = get_loaders_geoGuessr(
        train_config.batch_size, directory=train_config.dataset_dir
    )

    print("Training on device:", device)
    test_metrics, all_test_distances = train(
        config=train_config,
        net=net,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader
    )
    wandb.log(
        {
            "test/loss_avg": test_metrics.get("loss_avg", -1),
            "test/loss_std": test_metrics.get("loss_std", -1),
            "test/distance_avg": test_metrics.get("distance_avg", -1),
            "test/distance_std": test_metrics.get("distance_std", -1),
            "test/distance_p80": test_metrics.get("distance_p80", -1),
            "test/distance_p90": test_metrics.get("distance_p90", -1),
            "test/distance_p95": test_metrics.get("distance_p95", -1),
            "test/score_avg": test_metrics.get("score_avg", -1),
            "test/score_std": test_metrics.get("score_std", -1),
            "test/score_p80": test_metrics.get("score_p80", -1),
            "test/score_p90": test_metrics.get("score_p90", -1),
            "test/score_p95": test_metrics.get("score_p95", -1),
            "test/abs_err_lon_deg": test_metrics.get("abs_err_lon_deg", -1),
            "test/abs_err_lat_deg": test_metrics.get("abs_err_lat_deg", -1),
            "test/pred_lon_std": test_metrics.get("pred_lon_std", -1),
            "test/true_lon_std": test_metrics.get("true_lon_std", -1),
            "test/pred_lat_std": test_metrics.get("pred_lat_std", -1),
            "test/true_lat_std": test_metrics.get("true_lat_std", -1),
            "test/distance_histogram": wandb.Histogram(all_test_distances),
        }
    )