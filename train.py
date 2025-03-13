import os
import torch
from torch.utils.tensorboard import SummaryWriter

EARTH_RADIUS = 6371000


def get_logging_dir(directory="runs/"):
    os.makedirs(directory, exist_ok=True)
    existing_runs = [int(d) for d in os.listdir(directory) if d.isdigit()]
    next_run = max(existing_runs, default=-1) + 1
    run_dir = os.path.join(directory, str(next_run))

    return run_dir


def geoguessr_loss(pred, truth):
    # Calculate distance
    phi_pred = torch.deg2rad(pred[:, 0])  # B, 1
    phi_truth = torch.deg2rad(truth[:, 0])  # B, 1
    delta_phi = torch.deg2rad(truth[:, 0] - pred[:, 0])  # B, 1
    delta_lambda = torch.deg2rad(truth[:, 1] - pred[:, 1])  # B, 1
    a = torch.sin(delta_phi / 2.0)**2 + torch.cos(phi_pred) * torch.cos(phi_truth) * torch.sin(delta_lambda / 2.0)**2  # B, 1
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # B, 1
    distance = EARTH_RADIUS * c  # B, 1

    # Calculate GeoGuessr score
    scaling_factor = 2000
    score = 5000 * torch.exp(-distance / scaling_factor)

    return torch.mean(-score)


def evaluate(net, loader, loss_fn):
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in loader:
            out_val = net(X_val)
            loss_val = loss_fn(out_val, y_val)
            val_loss += loss_val.item()
    val_loss /= len(loader)

    return val_loss


def train(
    net, optimizer, epochs, train_loader, eval_loader, test_loader, loss_fn, writer
):
    for e in range(epochs):
        global_step = e * len(train_loader.dataset)  # Num training examples
        net.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            out = net(X)  # B, 2
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            global_step += X.size(0)
            writer.add_scalar("Loss/train", loss.item(), global_step)

        val_loss = evaluate(net, eval_loader, loss_fn)
        writer.add_scalar("Loss/validation", val_loss, e)

    return evaluate(net, test_loader, loss_fn)


if __name__ == "__main__":
    lr = 0.004
    bs= 64

    hparams = {"learning_rate": lr, "batch_size": bs}
    writer = SummaryWriter(get_logging_dir("runs/"))
    writer.close()

    test_loss = train(net, optimizer, 10, train_loader, eval_loader, geoguessr_loss, writer)

    writer.add_hparams(hparams, {"test_loss": test_loss})
    writer.close()
