import pandas as pd

from src.utils.config import caminho_absoluto


def carregar_splits(cultura, cfg):
    caminho = caminho_absoluto(cfg["dados"]["processed_dir"]) / cultura / "splits.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode: python -m src.data.preparar --cultura {cultura}"
        )
    return pd.read_csv(caminho)
