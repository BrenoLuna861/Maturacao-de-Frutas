"""Baseline classico: histograma HSV + SVM.

Termo de comparacao para a CNN. Roda em CPU, em segundos.

    python -m src.baseline --cultura manga
"""
import argparse
import json

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.data.splits import carregar_splits
from src.utils.config import caminho_absoluto, carregar_config, fixar_seed
from src.utils.metricas import calcular_metricas, plotar_matriz_confusao
from src.utils.registro import registrar

BINS = (16, 8, 8)


def extrair_features(caminho):
    imagem = cv2.imread(str(caminho_absoluto(caminho)))
    if imagem is None:
        return None

    imagem = cv2.resize(imagem, (256, 256))
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def montar_matriz(df, descricao):
    X, y, ilegiveis = [], [], []

    for caminho, rotulo in tqdm(
        zip(df["caminho"], df["rotulo"]), total=len(df), desc=descricao
    ):
        features = extrair_features(caminho)
        if features is None:
            ilegiveis.append(caminho)
            continue
        X.append(features)
        y.append(rotulo)

    if ilegiveis:
        print(f"  {len(ilegiveis)} imagem(ns) ilegivel(eis) ignorada(s): {ilegiveis[:3]}")
    if not X:
        raise ValueError(f"Nenhuma imagem legivel em '{descricao}'")

    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = carregar_config(args.config)
    fixar_seed(cfg["projeto"]["seed"])
    classes = cfg["dados"]["classes"]

    df = carregar_splits(args.cultura, cfg)
    X_treino, y_treino = montar_matriz(df.query("split == 'treino'"), "treino")
    X_teste, y_teste = montar_matriz(df.query("split == 'teste'"), "teste")

    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
            random_state=cfg["projeto"]["seed"]),
    )
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)

    m = calcular_metricas(y_teste, y_pred, classes)
    print(f"\nbaseline HSV+SVM | {args.cultura}")
    print(f"acuracia     {m['acuracia']:.3f}")
    print(f"recall macro {m['recall_macro']:.3f}")
    print(f"f1 macro     {m['f1_macro']:.3f}")

    fig = plotar_matriz_confusao(
        y_teste, y_pred, classes,
        caminho_absoluto(cfg["saida"]["figures_dir"]) / f"matriz_confusao_baseline_{args.cultura}.png",
        titulo=f"Baseline HSV+SVM - {args.cultura}",
    )
    print(fig)

    destino = caminho_absoluto(cfg["saida"]["models_dir"]) / f"{args.cultura}_baseline_metricas.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destino)

    csv = registrar(
        args.cultura, "baseline_hsv_svm", "teste", m, len(y_teste),
        observacao=f"bins={BINS}",
    )
    print(csv)


if __name__ == "__main__":
    main()
