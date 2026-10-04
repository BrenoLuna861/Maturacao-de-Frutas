import random
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[2]


def carregar_config(caminho="configs/config.yaml"):
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = RAIZ / caminho
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def caminho_absoluto(relativo):
    p = Path(relativo)
    return p if p.is_absolute() else RAIZ / p


def fixar_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
