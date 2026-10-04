"""Avalia o modelo treinado no conjunto de teste.

    python -m src.avaliar --cultura manga
"""
import argparse
import json

import pandas as pd
import torch

from src.data.dataset import criar_dataloaders
from src.models.construir import construir_modelo, escolher_dispositivo
from src.utils.config import caminho_absoluto, carregar_config, fixar_seed
from src.utils.metricas import calcular_metricas, plotar_matriz_confusao
from src.utils.registro import registrar


def prever(modelo, loader, dispositivo):
    modelo.eval()
    y_true, y_pred, confiancas = [], [], []

    with torch.no_grad():
        for imagens, rotulos in loader:
            probs = torch.softmax(modelo(imagens.to(dispositivo)), dim=1)
            conf, pred = probs.max(1)
            y_true += rotulos.tolist()
            y_pred += pred.cpu().tolist()
            confiancas += conf.cpu().tolist()

    return y_true, y_pred, confiancas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--experimento", default="", help="rotulo para identificar a rodada")
    ap.add_argument("--observacao", default="")
    args = ap.parse_args()

    cfg = carregar_config(args.config)
    fixar_seed(cfg["projeto"]["seed"])
    classes = cfg["dados"]["classes"]
    dispositivo = escolher_dispositivo()

    caminho_ckpt = caminho_absoluto(
        args.checkpoint or f"{cfg['saida']['models_dir']}/{args.cultura}.pt"
    )
    if not caminho_ckpt.exists():
        raise FileNotFoundError(
            f"{caminho_ckpt} nao existe. Rode: python -m src.treinar --cultura {args.cultura}"
        )

    modelo = construir_modelo(cfg["treino"]["arquitetura"], len(classes), False)
    ckpt = torch.load(caminho_ckpt, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(ckpt["estado"])
    modelo = modelo.to(dispositivo)

    loaders = criar_dataloaders(args.cultura, cfg)
    if "teste" not in loaders:
        raise ValueError("Nao ha split de teste.")

    y_true, y_pred, confiancas = prever(modelo, loaders["teste"], dispositivo)
    m = calcular_metricas(y_true, y_pred, classes)

    print(f"\n{args.cultura} | teste ({len(y_true)} imagens)")
    print(f"acuracia     {m['acuracia']:.3f}")
    print(f"recall macro {m['recall_macro']:.3f}")
    print(f"f1 macro     {m['f1_macro']:.3f}")
    print()
    for classe, valor in m["f1_por_classe"].items():
        print(f"  f1 {classe:<10} {valor:.3f}")

    dir_fig = caminho_absoluto(cfg["saida"]["figures_dir"])
    fig = plotar_matriz_confusao(
        y_true, y_pred, classes,
        dir_fig / f"matriz_confusao_{args.cultura}.png",
        titulo=f"Matriz de confusao - {args.cultura}",
    )
    print(f"\n{fig}")

    # erro com alta confianca costuma ser rotulo errado na base
    # ou atalho aprendido (fundo, iluminacao). vale olhar as imagens.
    splits = caminho_absoluto(cfg["dados"]["processed_dir"]) / args.cultura / "splits.csv"
    df = pd.read_csv(splits).query("split == 'teste'").reset_index(drop=True)
    df["predito"] = [classes[p] for p in y_pred]
    df["confianca"] = confiancas

    erros = df[df["classe"] != df["predito"]].sort_values("confianca", ascending=False)
    caminho_erros = dir_fig / f"erros_{args.cultura}.csv"
    erros.to_csv(caminho_erros, index=False)
    print(f"{len(erros)} erros -> {caminho_erros}")

    if len(erros):
        print()
        print(erros[["caminho", "classe", "predito", "confianca"]].head().to_string(index=False))

    destino = caminho_absoluto(cfg["saida"]["models_dir"]) / f"{args.cultura}_metricas.json"
    destino.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{destino}")

    csv = registrar(
        args.cultura, cfg["treino"]["arquitetura"], "teste", m, len(y_true),
        experimento=args.experimento or caminho_ckpt.stem,
        observacao=args.observacao,
    )
    print(csv)


if __name__ == "__main__":
    main()
