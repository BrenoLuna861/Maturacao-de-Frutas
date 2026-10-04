"""Monta a tabela baseline x CNN a partir de reports/experimentos.csv.

    python -m src.comparar --cultura manga
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import caminho_absoluto

METRICAS = ["acuracia", "recall_macro", "f1_macro"]
ROTULOS = {"acuracia": "Acuracia", "recall_macro": "Recall macro", "f1_macro": "F1 macro"}


def carregar(cultura=None):
    caminho = caminho_absoluto("reports") / "experimentos.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode o baseline e a avaliacao antes."
        )
    df = pd.read_csv(caminho)
    df = df[df["split"] == "teste"]
    if cultura:
        df = df[df["cultura"] == cultura]
    if df.empty:
        raise ValueError("Nenhum resultado de teste registrado ainda.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura")
    args = ap.parse_args()

    df = carregar(args.cultura)

    # fica o melhor f1 de cada modelo
    melhores = df.sort_values("f1_macro").groupby("modelo").tail(1)
    melhores = melhores.sort_values("f1_macro", ascending=False)

    print(melhores[["cultura", "modelo", "acuracia", "recall_macro", "f1_macro", "data"]]
          .to_string(index=False))

    titulo = f"Baseline x CNN - {args.cultura}" if args.cultura else "Baseline x CNN"
    modelos = melhores["modelo"].tolist()
    largura = 0.26
    posicoes = range(len(METRICAS))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, modelo in enumerate(modelos):
        linha = melhores[melhores["modelo"] == modelo].iloc[0]
        valores = [linha[m] for m in METRICAS]
        deslocadas = [p + i * largura for p in posicoes]
        barras = ax.bar(deslocadas, valores, largura, label=modelo)
        ax.bar_label(barras, fmt="%.3f", fontsize=8)

    ax.set_xticks([p + largura * (len(modelos) - 1) / 2 for p in posicoes])
    ax.set_xticklabels([ROTULOS[m] for m in METRICAS])
    ax.set_ylim(0, 1.1)
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()

    sufixo = f"_{args.cultura}" if args.cultura else ""
    destino = caminho_absoluto("reports/figures") / f"comparacao{sufixo}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"\n{destino}")


if __name__ == "__main__":
    main()
