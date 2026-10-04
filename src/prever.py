"""Classifica imagens novas com um modelo treinado.

    python -m src.prever --cultura manga --imagem foto.jpg
    python -m src.prever --cultura manga --pasta fotos/ --csv saida.csv
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageFile

from src.data.dataset import construir_transformacoes
from src.models.construir import construir_modelo, escolher_dispositivo
from src.utils.config import caminho_absoluto, carregar_config

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def carregar_modelo(caminho_ckpt, cfg, n_classes, dispositivo):
    modelo = construir_modelo(cfg["treino"]["arquitetura"], n_classes, False)
    ckpt = torch.load(caminho_ckpt, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(ckpt["estado"])
    return modelo.to(dispositivo).eval()


def prever_imagem(modelo, caminho, transformacao, classes, dispositivo):
    imagem = Image.open(caminho).convert("RGB")
    tensor = transformacao(imagem).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        probs = torch.softmax(modelo(tensor), dim=1)[0]

    return {
        "arquivo": str(caminho),
        "predito": classes[int(probs.argmax())],
        "confianca": float(probs.max()),
        "probabilidades": {c: float(p) for c, p in zip(classes, probs)},
    }


def listar_imagens(pasta):
    return sorted(p for p in Path(pasta).rglob("*") if p.suffix.lower() in EXTENSOES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura", required=True)
    ap.add_argument("--imagem", help="uma imagem")
    ap.add_argument("--pasta", help="todas as imagens de uma pasta")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--csv", help="salva o resultado neste arquivo")
    args = ap.parse_args()

    if not args.imagem and not args.pasta:
        ap.error("informe --imagem ou --pasta")

    cfg = carregar_config(args.config)
    classes = cfg["dados"]["classes"]
    dispositivo = escolher_dispositivo()

    caminho_ckpt = caminho_absoluto(
        args.checkpoint or f"{cfg['saida']['models_dir']}/{args.cultura}.pt"
    )
    if not caminho_ckpt.exists():
        raise FileNotFoundError(
            f"{caminho_ckpt} nao existe. Treine antes: python -m src.treinar --cultura {args.cultura}"
        )

    modelo = carregar_modelo(caminho_ckpt, cfg, len(classes), dispositivo)
    transformacao = construir_transformacoes(cfg, treino=False)

    alvos = [Path(args.imagem)] if args.imagem else listar_imagens(args.pasta)
    if not alvos:
        raise ValueError(f"Nenhuma imagem encontrada em {args.pasta}")

    resultados = []
    for alvo in alvos:
        try:
            r = prever_imagem(modelo, alvo, transformacao, classes, dispositivo)
        except OSError as e:
            print(f"[ilegivel] {alvo}: {e}")
            continue

        resultados.append(r)
        detalhe = "  ".join(f"{c} {p:.2f}" for c, p in r["probabilidades"].items())
        print(f"{alvo.name:<40} {r['predito']:<10} {r['confianca']:.1%}   [{detalhe}]")

    if args.csv:
        destino = Path(args.csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arquivo", "predito", "confianca", *classes])
            for r in resultados:
                w.writerow([
                    r["arquivo"], r["predito"], round(r["confianca"], 4),
                    *[round(r["probabilidades"][c], 4) for c in classes],
                ])
        print(f"\n{destino}")

    if len(resultados) > 1:
        from collections import Counter
        contagem = Counter(r["predito"] for r in resultados)
        print(f"\n{len(resultados)} imagens: " +
              ", ".join(f"{c} {n}" for c, n in contagem.most_common()))


if __name__ == "__main__":
    main()
