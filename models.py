import torch
import torchvision

NUM_CLASSES = 2


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


def get_net(net_name="resnet50", dropout_rate=0.5, device="cpu"):
    if "resnet" in net_name:
        return get_resnet(net_name, dropout_rate).to(device)
    elif "vit" in net_name:
        return get_vit(net_name, dropout_rate).to(device)
    elif "efficientnet" in net_name:
        return get_efficientnet(net_name, dropout_rate).to(device)
    else:
        print(f"{net_name} not supported")
        exit(0)
