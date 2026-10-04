"""Gera imagens sinteticas para testar o pipeline sem a base real.

Sao manchas coloridas, nao servem para pesquisa. Servem para conferir que
preparar -> treinar -> avaliar roda antes de baixar os datasets.

    python scripts/gerar_dados_sinteticos.py --cultura manga --por-classe 30
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]

CORES = {
    "verde": (60, 140, 60),
    "de_vez": (200, 170, 50),
    "maduro": (200, 70, 40),
}


def gerar(cultura, por_classe, frutos_por_grupo, seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)

    for classe, cor_base in CORES.items():
        destino = RAIZ / "data" / "raw" / cultura / classe
        destino.mkdir(parents=True, exist_ok=True)
        n_grupos = max(1, por_classe // frutos_por_grupo)

        for grupo in range(n_grupos):
            desvio = rng.normal(0, 18, 3)  # cor propria de cada "fruto"
            for foto in range(frutos_por_grupo):
                cor = np.clip(np.array(cor_base) + desvio, 0, 255)
                img = np.clip(cor + rng.normal(0, 10, (128, 128, 3)), 0, 255).astype(np.uint8)
                nome = f"{cultura}_{classe}_{grupo:03d}_{chr(97 + foto)}.jpg"
                Image.fromarray(img).save(destino / nome, quality=90)

        print(f"{classe}: {n_grupos * frutos_por_grupo} imagens em {n_grupos} grupos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura", default="manga")
    ap.add_argument("--por-classe", type=int, default=30)
    ap.add_argument("--frutos-por-grupo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gerar(args.cultura, args.por_classe, args.frutos_por_grupo, args.seed)
    print(f"\nagora: python -m src.data.preparar --cultura {args.cultura}")


if __name__ == "__main__":
    main()
