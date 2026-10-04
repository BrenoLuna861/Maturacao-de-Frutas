"""Treina o classificador de estagio de maturacao.

    python -m src.treinar --cultura manga
    python -m src.treinar --cultura manga --kfold
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import (
    DatasetFrutos,
    construir_transformacoes,
    criar_dataloaders,
    pesos_das_classes,
)
from src.data.splits import carregar_splits
from src.models.construir import congelar_backbone, construir_modelo, escolher_dispositivo
from src.utils.config import caminho_absoluto, carregar_config, fixar_seed
from src.utils.metricas import plotar_curvas


def _uma_epoca(modelo, loader, criterio, otimizador, dispositivo, treinando):
    modelo.train() if treinando else modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.set_grad_enabled(treinando):
        for imagens, rotulos in loader:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            if treinando:
                otimizador.zero_grad()
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)
            if treinando:
                perda.backward()
                otimizador.step()

            perda_total += perda.item() * imagens.size(0)
            acertos += (saidas.argmax(1) == rotulos).sum().item()
            total += imagens.size(0)

    return perda_total / max(total, 1), acertos / max(total, 1)


def treinar(modelo, loaders, cfg, dispositivo, pesos_classe=None, tag="modelo"):
    t = cfg["treino"]
    criterio = nn.CrossEntropyLoss(
        weight=pesos_classe.to(dispositivo) if pesos_classe is not None else None
    )

    historico = {"loss_treino": [], "acc_treino": [], "loss_val": [], "acc_val": []}
    melhor_acc = 0.0
    sem_melhora = 0

    dir_modelos = caminho_absoluto(cfg["saida"]["models_dir"])
    dir_modelos.mkdir(parents=True, exist_ok=True)
    caminho_melhor = dir_modelos / f"{tag}.pt"

    # fase 1: so a cabeca. fase 2: tudo liberado com lr menor.
    fases = [
        ("cabeca", t["epocas_cabeca"], t["lr_cabeca"], True),
        ("finetune", t["epocas_finetune"], t["lr_finetune"], False),
    ]

    for nome_fase, n_epocas, lr, congelar in fases:
        if n_epocas <= 0:
            continue

        congelar_backbone(modelo, congelar)
        otimizador = torch.optim.AdamW(
            [p for p in modelo.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=t["weight_decay"],
        )
        print(f"\nfase {nome_fase} (lr={lr})")

        for epoca in tqdm(range(1, n_epocas + 1), desc=nome_fase):
            lt, at = _uma_epoca(modelo, loaders["treino"], criterio, otimizador, dispositivo, True)
            lv, av = _uma_epoca(modelo, loaders["validacao"], criterio, None, dispositivo, False)

            historico["loss_treino"].append(lt)
            historico["acc_treino"].append(at)
            historico["loss_val"].append(lv)
            historico["acc_val"].append(av)
            print(f"  epoca {epoca}: treino {lt:.4f}/{at:.3f} | val {lv:.4f}/{av:.3f}")

            if av > melhor_acc:
                melhor_acc = av
                sem_melhora = 0
                torch.save({"estado": modelo.state_dict(), "acc_val": av, "config": cfg}, caminho_melhor)
            else:
                sem_melhora += 1
                if sem_melhora >= t["paciencia_early_stopping"]:
                    print(f"  parando: {sem_melhora} epocas sem melhora")
                    break

    print(f"\nmelhor acuracia de validacao: {melhor_acc:.3f} -> {caminho_melhor}")
    return historico, melhor_acc, caminho_melhor


def _validacao_cruzada(cultura, cfg, dispositivo, classes, pre_treinado=True):
    k = cfg["validacao_cruzada"]["k"]
    df = carregar_splits(cultura, cfg)
    df = df[df["split"] != "teste"].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["projeto"]["seed"])
    resultados = []

    for fold, (idx_tr, idx_va) in enumerate(sgkf.split(df, df["rotulo"], groups=df["grupo"]), 1):
        print(f"\n##### fold {fold}/{k} #####")
        loaders = {
            "treino": DataLoader(
                DatasetFrutos(df.iloc[idx_tr], construir_transformacoes(cfg, True)),
                batch_size=cfg["treino"]["batch_size"],
                shuffle=True,
                num_workers=cfg["treino"]["num_workers"],
            ),
            "validacao": DataLoader(
                DatasetFrutos(df.iloc[idx_va], construir_transformacoes(cfg, False)),
                batch_size=cfg["treino"]["batch_size"],
                num_workers=cfg["treino"]["num_workers"],
            ),
        }
        modelo = construir_modelo(cfg["treino"]["arquitetura"], len(classes), pre_treinado)
        modelo = modelo.to(dispositivo)
        _, acc, _ = treinar(
            modelo, loaders, cfg, dispositivo,
            pesos_das_classes(df.iloc[idx_tr], len(classes)),
            tag=f"{cultura}_fold{fold}",
        )
        resultados.append(acc)

    media = float(np.mean(resultados))
    desvio = float(np.std(resultados))
    print(f"\nvalidacao cruzada ({k} folds): {media:.3f} +/- {desvio:.3f}")
    print(f"folds: {[round(r, 3) for r in resultados]}")

    destino = caminho_absoluto(cfg["saida"]["models_dir"]) / f"{cultura}_cv.json"
    destino.write_text(
        json.dumps({"k": k, "acuracias": resultados, "media": media, "desvio": desvio}, indent=2),
        encoding="utf-8",
    )
    print(destino)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--kfold", action="store_true")
    ap.add_argument("--sem-pre-treino", action="store_true",
                    help="nao baixa os pesos do ImageNet (so para teste offline)")
    args = ap.parse_args()

    cfg = carregar_config(args.config)
    fixar_seed(cfg["projeto"]["seed"])
    dispositivo = escolher_dispositivo()
    classes = cfg["dados"]["classes"]
    pre_treinado = not args.sem_pre_treino

    print(f"dispositivo: {dispositivo}")
    if not pre_treinado:
        print("rodando sem pesos do ImageNet")

    if args.kfold or cfg["validacao_cruzada"]["ativa"]:
        _validacao_cruzada(args.cultura, cfg, dispositivo, classes, pre_treinado)
        return

    loaders = criar_dataloaders(args.cultura, cfg)
    df_treino = carregar_splits(args.cultura, cfg).query("split == 'treino'")
    modelo = construir_modelo(cfg["treino"]["arquitetura"], len(classes), pre_treinado)
    modelo = modelo.to(dispositivo)

    inicio = time.time()
    historico, _, _ = treinar(
        modelo, loaders, cfg, dispositivo,
        pesos_das_classes(df_treino, len(classes)),
        tag=args.cultura,
    )
    print(f"tempo: {(time.time() - inicio) / 60:.1f} min")

    fig = plotar_curvas(
        historico,
        caminho_absoluto(cfg["saida"]["figures_dir"]) / f"curvas_{args.cultura}.png",
    )
    print(fig)
    print(f"\nagora: python -m src.avaliar --cultura {args.cultura}")


if __name__ == "__main__":
    main()
