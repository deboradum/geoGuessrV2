import os
import torch
import time
import wandb
import torchvision

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import GeoGuessrDataset

EARTH_RADIUS = 6371000
MAX_DISTANCE = 20000000.0
NUM_CLASSES = 2

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def get_resnet(resnet, dropout_rate):
    if resnet == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        resnet_model = torchvision.models.resnet18
    elif resnet == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        resnet_model = torchvision.models.resnet34
    elif resnet == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet_model = torchvision.models.resnet50
    elif resnet == "resnet101":
        weights = torchvision.models.ResNet101_Weights.DEFAULT
        resnet_model = torchvision.models.resnet101
    elif resnet == "resnet152":
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        resnet_model = torchvision.models.resnet152
    else:
        print(f"{resnet} not supported")
        exit(0)

    net = resnet_model(weights=weights)

    net.fc = torch.nn.Sequential(
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(net.fc.in_features, NUM_CLASSES),
    )
    torch.nn.init.kaiming_uniform_(net.fc[1].weight)

    return net


def get_vit(net_name, dropout_rate):
    if net_name == "vit_b_16":
        weights = torchvision.models.vision_transformer.ViT_B_16_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_b_16
    elif net_name == "vit_b_32":
        weights = torchvision.models.vision_transformer.ViT_B_32_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_b_32
    elif net_name == "vit_l_16":
        weights = torchvision.models.vision_transformer.ViT_L_16_Weights.DEFAULT
        vit_model = torchvision.models.vision_transformer.vit_l_16
    else:
        print(f"{net_name} not supported")
        exit(0)

    net = vit_model(weights=weights)

    net.heads.head = torch.nn.Sequential(
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(net.heads.head.in_features, NUM_CLASSES),
    )
    torch.nn.init.kaiming_uniform_(net.heads.head[1].weight)

    return net


def get_efficientnet(net_name, dropout_rate):
    if net_name == "efficientnet_b0":
        weights = torchvision.models.efficientnet.EfficientNet_B0_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b0
    elif net_name == "efficientnet_b1":
        weights = torchvision.models.efficientnet.EfficientNet_B1_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b1
    elif net_name == "efficientnet_b2":
        weights = torchvision.models.efficientnet.EfficientNet_B2_Weights.DEFAULT
        efficientnet_model = torchvision.models.efficientnet_b2
    else:
        print(f"{net_name} not supported")
        exit(0)

    net = efficientnet_model(weights=weights)

    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, NUM_CLASSES)
    net.classifier.add_module("dropout", torch.nn.Dropout(dropout_rate))
    torch.nn.init.kaiming_uniform_(net.classifier[1].weight)

    return net


def get_net(net_name="resnet50", dropout_rate=0.5):
    if "resnet" in net_name:
        return get_resnet(net_name, dropout_rate).to(device)
    elif "vit" in net_name:
        return get_vit(net_name, dropout_rate).to(device)
    elif "efficientnet" in net_name:
        return get_efficientnet(net_name, dropout_rate).to(device)
    else:
        print(f"{net_name} not supported")
        exit(0)


def get_loaders(batch_size, model_name, directory="createDataset/dataset/"):
    transform = transforms.Compose(
        [
            transforms.Resize((240, 240))
            if "efficientnet" in model_name
            else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    datasets = {
        "train": GeoGuessrDataset(
            os.path.join(directory, "train.csv"), directory, transform
        ),
        "val": GeoGuessrDataset(
            os.path.join(directory, "val.csv"), directory, transform
        ),
        "test": GeoGuessrDataset(
            os.path.join(directory, "test.csv"), directory, transform
        ),
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=4,
        )
        for split in ["train", "val", "test"]
    }

    return loaders["train"], loaders["val"], loaders["test"]


def get_logging_dir(directory="runs/"):
    os.makedirs(directory, exist_ok=True)
    existing_runs = [int(d) for d in os.listdir(directory) if d.isdigit()]
    next_run = max(existing_runs, default=-1) + 1
    run_dir = os.path.join(directory, str(next_run))

    return run_dir


def geoguessr_loss(pred, truth):
    # Calculate distance
    #phi_pred = torch.deg2rad(pred[:, 0])  # B, 1
    #phi_truth = torch.deg2rad(truth[:, 0])  # B, 1
    #delta_phi = torch.deg2rad(truth[:, 0] - pred[:, 0])  # B, 1
    #delta_lambda = torch.deg2rad(truth[:, 1] - pred[:, 1])  # B, 1
    #a = (
    #    torch.sin(delta_phi / 2.0) ** 2
    #    + torch.cos(phi_pred)
    #    * torch.cos(phi_truth)
    #    * torch.sin(delta_lambda / 2.0) ** 2
    #)  # B, 1
    #c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # B, 1
    #distance = EARTH_RADIUS * c  # B, 1

    # Loss v1
    # Calculate GeoGuessr score
    #scaling_factor = 2000000
    #score = 5000 * torch.exp(-distance / scaling_factor)
    #loss = torch.mean(-torch.log(score + 1e-9))

    # Loss v2
    #loss = distance.mean()

    # Loss v3
    #log_distance = torch.log1p(distance)
    #max_log_distance = torch.log1p(torch.tensor(MAX_DISTANCE, device=device))
    #log_distance_normalized = log_distance / max_log_distance
    #loss = log_distance_normalized.mean()

    # Loss V4
    loss = F.mse_loss(pred, truth, reduction="none").mean(dim=1).mean()

    return loss


def evaluate(net, loader, loss_fn):
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = net(X)
            out_lat = out[:, 0] * 90  # Scale latitude to [-90, 90]
            out_lon = out[:, 1] * 180  # Scale longitude to [-180, 180]
            out_scaled = torch.stack([out_lat, out_lon], dim=1)
            loss = loss_fn(out_scaled, y)
            val_loss += loss.item()
    val_loss /= len(loader)

    return val_loss


def train(net, optimizer, epochs, train_loader, eval_loader, test_loader, loss_fn):
    val_loss = evaluate(net, eval_loader, loss_fn)
    wandb.log({"eval_loss": val_loss})
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
            wandb.log({"train_loss": loss.item(), "step": global_step})
        val_loss = evaluate(net, eval_loader, loss_fn)
        took = round(time.time() - s, 3)
        print(f"Epoch {e} finished, val loss: {round(val_loss, 4)}, took {took}s")
        wandb.log({"eval_loss": val_loss, "epoch": e})

    return evaluate(net, test_loader, loss_fn)


def wandb_train():
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

    net = get_net(net_name, dropout)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optim == "adamW":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise Exception("Invalid optimizer")

    train_loader, eval_loader, test_loader = get_loaders(
        bs, net_name, directory="createDataset/dataset/"
    )

    test_loss = train(
        net, optimizer, epochs, train_loader, eval_loader, test_loader, geoguessr_loss
    )
    wandb.log({"test_loss": test_loss})


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
