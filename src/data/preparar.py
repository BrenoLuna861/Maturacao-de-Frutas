"""Gera os splits de treino/validacao/teste a partir de data/raw.

Entrada:  data/raw/<cultura>/<classe>/<arquivo>.jpg
Saida:    data/processed/<cultura>/splits.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import caminho_absoluto, carregar_config, fixar_seed

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalizar_classe(nome, mapeamento, classes):
    chave = nome.strip().lower().replace(" ", "_").replace("-", "_")
    if chave in classes:
        return chave
    return mapeamento.get(chave)


def _extrair_grupo(nome_arquivo, regex):
    m = re.match(regex, nome_arquivo)
    return m.group(1) if m else Path(nome_arquivo).stem


def indexar(cultura, cfg):
    raw = caminho_absoluto(cfg["dados"]["raw_dir"]) / cultura
    if not raw.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {raw}")

    classes = cfg["dados"]["classes"]
    mapeamento = cfg["dados"].get("mapeamento_classes") or {}
    regex = cfg["dados"]["regex_grupo"]
    base = caminho_absoluto(".")

    linhas = []
    ignoradas = []

    for pasta in sorted(p for p in raw.iterdir() if p.is_dir()):
        classe = _normalizar_classe(pasta.name, mapeamento, classes)
        if classe is None:
            ignoradas.append(pasta.name)
            continue

        for img in sorted(pasta.rglob("*")):
            if img.suffix.lower() not in EXTENSOES:
                continue
            linhas.append({
                "caminho": str(img.relative_to(base)).replace("\\", "/"),
                "classe": classe,
                "rotulo": classes.index(classe),
                "grupo": f"{cultura}/{_extrair_grupo(img.name, regex)}",
            })

    if ignoradas:
        print(f"  pastas ignoradas: {', '.join(ignoradas)}")
        print("  (mapeie esses nomes em mapeamento_classes no config)")
    if not linhas:
        raise ValueError(f"Nenhuma imagem valida em {raw}")

    return pd.DataFrame(linhas)


def dividir(df, cfg):
    p = cfg["dados"]["split"]
    seed = cfg["projeto"]["seed"]

    # separa o teste
    n = max(2, round(1 / p["teste"]))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    idx_resto, idx_teste = next(sgkf.split(df, df["rotulo"], groups=df["grupo"]))

    # separa a validacao do que sobrou
    resto = df.iloc[idx_resto].reset_index(drop=True)
    frac_val = p["validacao"] / (p["treino"] + p["validacao"])
    n = max(2, round(1 / frac_val))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    idx_treino, idx_val = next(sgkf.split(resto, resto["rotulo"], groups=resto["grupo"]))

    df = df.copy()
    df["split"] = ""
    df.loc[df.index[idx_teste], "split"] = "teste"
    df.loc[df["caminho"].isin(set(resto.iloc[idx_treino]["caminho"])), "split"] = "treino"
    df.loc[df["caminho"].isin(set(resto.iloc[idx_val]["caminho"])), "split"] = "validacao"

    _conferir_vazamento(df)
    return df


def _conferir_vazamento(df):
    """Falha se o mesmo fruto aparecer em mais de um split."""
    por_grupo = df.groupby("grupo")["split"].nunique()
    vazados = por_grupo[por_grupo > 1]
    if len(vazados):
        raise AssertionError(
            f"{len(vazados)} grupo(s) em mais de um split: {list(vazados.index[:5])}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultura")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = carregar_config(args.config)
    fixar_seed(cfg["projeto"]["seed"])
    culturas = [args.cultura] if args.cultura else cfg["dados"]["culturas"]

    for cultura in culturas:
        print(f"\n=== {cultura} ===")
        df = dividir(indexar(cultura, cfg), cfg)

        destino = caminho_absoluto(cfg["dados"]["processed_dir"]) / cultura
        destino.mkdir(parents=True, exist_ok=True)
        df.to_csv(destino / "splits.csv", index=False)

        print(f"{len(df)} imagens, {df['grupo'].nunique()} grupos")
        print(pd.crosstab(df["split"], df["classe"]))
        print(destino / "splits.csv")


if __name__ == "__main__":
    main()
