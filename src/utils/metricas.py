from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)


def calcular_metricas(y_true, y_pred, classes):
    f1_classes = f1_score(
        y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0
    )
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_por_classe": {c: float(v) for c, v in zip(classes, f1_classes)},
        "relatorio": classification_report(
            y_true, y_pred, target_names=classes, zero_division=0, output_dict=True
        ),
    }


def plotar_matriz_confusao(y_true, y_pred, classes, destino, titulo="Matriz de confusao"):
    mc = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        mc_norm = np.nan_to_num(mc / mc.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 1.4 * len(classes) + 2))
    im = ax.imshow(mc_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    limite = mc_norm.max() / 2 if mc_norm.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            cor = "white" if mc_norm[i, j] > limite else "black"
            ax.text(j, i, f"{mc[i, j]}\n{mc_norm[i, j]:.0%}",
                    ha="center", va="center", color=cor, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


def plotar_curvas(historico, destino):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epocas = range(1, len(historico["loss_treino"]) + 1)

    ax1.plot(epocas, historico["loss_treino"], label="treino")
    ax1.plot(epocas, historico["loss_val"], label="validacao")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Perda")
    ax1.legend()

    ax2.plot(epocas, historico["acc_treino"], label="treino")
    ax2.plot(epocas, historico["acc_val"], label="validacao")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia")
    ax2.legend()

    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino
