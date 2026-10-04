"""Registro de experimentos em reports/experimentos.csv."""
import csv
from datetime import datetime

from src.utils.config import caminho_absoluto

COLUNAS = [
    "data", "experimento", "cultura", "modelo", "split",
    "acuracia", "recall_macro", "f1_macro", "n_imagens", "observacao",
]


def nome_experimento(prefixo):
    return f"{prefixo}_{datetime.now():%Y%m%d_%H%M%S}"


def registrar(cultura, modelo, split, metricas, n_imagens, experimento="", observacao=""):
    """Acrescenta uma linha ao CSV de experimentos. Nunca sobrescreve."""
    destino = caminho_absoluto("reports") / "experimentos.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    novo = not destino.exists()

    with open(destino, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            w.writeheader()
        w.writerow({
            "data": datetime.now().isoformat(timespec="seconds"),
            "experimento": experimento,
            "cultura": cultura,
            "modelo": modelo,
            "split": split,
            "acuracia": round(metricas["acuracia"], 4),
            "recall_macro": round(metricas["recall_macro"], 4),
            "f1_macro": round(metricas["f1_macro"], 4),
            "n_imagens": n_imagens,
            "observacao": observacao,
        })

    return destino
