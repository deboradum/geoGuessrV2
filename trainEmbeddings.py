import time
import torch
import wandb
import argparse

from pytorch_metric_learning import miners, losses

from models import get_net
from dataset import get_loaders_geoGuessrEmbedding
from utils import TrainConfig, load_config, get_optimizer

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def evaluate(net, loader):
    loss_fn = losses.TripletMarginLoss()
    miner_fn = miners.BatchHardMiner()
    total_samples = 0
    total_loss = 0.

    net.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            embeddings = net(X, embedding_only=True)
            miner_output = miner_fn(embeddings, y)

            loss = loss_fn(embeddings, y, miner_output)
            bs = X.size(0)

            total_loss += loss * bs
            total_samples += bs

    if total_samples == 0:
        return -1

    return total_loss / total_samples


def train(
    config: TrainConfig,
    net,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader
):
    global_step = 0

    # Evaluate
    start = time.perf_counter()
    net.eval()
    val_loss = evaluate(net, eval_loader)
    net.train()
    taken = time.perf_counter() - start
    wandb.log(
        {
            "epoch": 0,
            "examples": global_step,
            "eval/loss": val_loss,
        }
    )
    print(f"[Eval] Epoch 0, Time: {taken:.2f}s, loss: {val_loss:.3f}")

    loss_fn = losses.TripletMarginLoss()
    miner_fn = miners.BatchHardMiner()

    start = time.perf_counter()
    for e in range(config.epochs):
        total_loss = 0.
        total_grad_norm_before = 0.
        total_grad_norm_after = 0.

        net.train()
        optimizer.zero_grad()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            embeddings = net(X, embedding_only=True)
            miner_output = miner_fn(embeddings, y)

            loss = loss_fn(embeddings, y, miner_output)
            total_loss += loss.item()

            loss.backward()

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

                l = total_loss / config.log_interval

                wandb.log(
                    {
                        "epoch": e,
                        "batch": i,
                        "train/iterations_per_second": ips,
                        "examples": global_step,
                        "train/loss": l,
                        "optimizer/grad_norm_before_clip": total_grad_norm_before / config.log_interval,
                        "optimizer/grad_norm_after_clip": total_grad_norm_after / config.log_interval,
                    }
                )
                print(f"Epoch {e}, step {i} (Global {global_step}), Time: {taken:.2f}s ({ips:.2f} i/s), loss: {l:.3f}")

                total_loss = 0.
                total_grad_norm_before = 0.
                total_grad_norm_after = 0.
                start = time.perf_counter()

        # Evaluate
        start = time.perf_counter()
        net.eval()
        val_loss = evaluate(net, eval_loader)
        net.train()
        taken = time.perf_counter() - start
        wandb.log(
            {
                "epoch": e+1,
                "examples": global_step,
                "eval/loss": val_loss,
            }
        )
        print(f"[Eval] Epoch {e+1}, Time: {taken:.2f}s, loss: {val_loss:.3f}")
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
    train_config = load_config(args.config)
    config_dict = {**vars(train_config)}

    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Setting up model")
    net = get_net(config=train_config, device=device)
    optimizer = get_optimizer(train_config, net)

    if args.compile:
        print("Compiling network")
        net = torch.compile(net)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params
    size = train_config.net_name.split("-")[-1]
    wandb.init(project="GeoGuessrCoordinatesV2-metricModel", name=train_config.run_name, config=config_dict, tags=[size])
    wandb.watch(net, log="all", log_freq=train_config.log_interval * 10) # Log grads & params

    train_loader, eval_loader, test_loader = get_loaders_geoGuessrEmbedding(
        directory=train_config.dataset_dir,
        s2_cell_level=train_config.s2_cell_level,
    )

    print("Training on device:", device)
    test_loss = train(
        config=train_config,
        net=net,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader
    )
    wandb.log({"test/loss": test_loss})

    save_path = f"{train_config.run_name}.pth"
    torch.save(net.state_dict(), save_path)
