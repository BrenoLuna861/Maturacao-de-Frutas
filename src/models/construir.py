import torch
import torch.nn as nn
from torchvision import models

ARQUITETURAS = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    ),
}


def construir_modelo(arquitetura, n_classes, pesos_pre_treinados=True):
    if arquitetura not in ARQUITETURAS:
        raise ValueError(f"Arquitetura desconhecida: {arquitetura}")

    construtor, pesos = ARQUITETURAS[arquitetura]
    modelo = construtor(weights=pesos if pesos_pre_treinados else None)

    # troca a ultima camada pelo nosso numero de classes
    n_entradas = modelo.classifier[-1].in_features
    modelo.classifier[-1] = nn.Linear(n_entradas, n_classes)
    return modelo


def congelar_backbone(modelo, congelar=True):
    for p in modelo.features.parameters():
        p.requires_grad = not congelar
    return modelo


def escolher_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
