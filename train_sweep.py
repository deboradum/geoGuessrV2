import torch
import wandb

from models import get_net
from train import get_args, train
from dataset import get_loaders
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
        "metric": {"name": "test/distance_avg", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.00005, "max": 0.0005},
            "gradient_clipping_norm": {"values": [0.0, 1.0, 2.0, 3.0]},
            "num_experts": {"values": [8, 16, 32]},
            "s2_loss_weight": {"min": 0.05, "max": 0.25},
            "load_balance_loss_weight": {"min": 0.05, "max": 0.25},
            "s2_cell_level": {"values": [4, 5, 7, 8]},
        },
    }

    def sweep_train():
        wandb.init(project="GeoGuessrCoordinates", name=train_config.run_name)
        config = wandb.config

        # update configs with sweep params
        train_config.learning_rate = config.learning_rate
        train_config.gradient_clipping_norm = config.gradient_clipping_norm
        train_config.num_experts = config.num_experts
        train_config.s2_loss_weight = config.s2_loss_weight
        train_config.load_balance_loss_weight = config.load_balance_loss_weight
        train_config.s2_cell_level = config.s2_cell_level

        config_dict = {**vars(train_config)}

        train_loader, eval_loader, test_loader = get_loaders(
            directory=train_config.dataset_dir, s2_cell_level=train_config.s2_cell_level,
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
        wandb.config.update(config_dict, allow_val_change=True)

        print("Training on device:", device)
        test_metrics, all_test_distances = train(
            config=train_config,
            net=net,  # type: ignore
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader
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
        wandb.finish()

    # sweep_id = wandb.sweep(sweep_config, project="GeoGuessrCoordinates")
    sweep_id = "xlelcbc7"
    wandb.agent(sweep_id, function=sweep_train, project="GeoGuessrCoordinates")
