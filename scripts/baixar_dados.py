"""Baixa bases publicas de maturacao via Kaggle e organiza em data/raw/.

Precisa da credencial do Kaggle: em kaggle.com -> Settings -> Create New Token,
salve o kaggle.json em %USERPROFILE%\\.kaggle\\ (Windows) ou ~/.kaggle/ (Linux).

    python scripts/baixar_dados.py --listar
    python scripts/baixar_dados.py --base manga_srabon

Depois de baixar, confira os nomes das pastas de classe e ajuste
`mapeamento_classes` em configs/config.yaml antes de rodar o preparar.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# cultura, slug do kaggle. conferidos em out/2026; slugs mudam, confira antes.
BASES = {
    "manga_srabon": ("manga", "srabon00/mango-ripening-stage-classification"),
    "manga_denchai": ("manga", "denchai/classification-of-ripeness-stage-of-mango-fruit"),
    "frutas_geral": ("manga", "asadullahprl/fruits-ripeness-classification-dataset"),
}


def kaggle_disponivel():
    return shutil.which("kaggle") is not None


def baixar(chave):
    if chave not in BASES:
        print(f"Base desconhecida: {chave}")
        print("Opcoes:", ", ".join(BASES))
        return 1

    cultura, slug = BASES[chave]
    destino = RAIZ / "data" / "raw" / f"_download_{chave}"
    destino.mkdir(parents=True, exist_ok=True)

    print(f"baixando {slug} -> {destino}")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(destino), "--unzip"],
        check=False,
    )
    if r.returncode != 0:
        print("\nFalhou. Causas comuns: kaggle.json ausente, ou o dataset exige")
        print("aceitar os termos uma vez pelo site antes de baixar.")
        return r.returncode

    print(f"\nPronto. Agora mova as pastas de classe para data/raw/{cultura}/")
    print("e confira os nomes delas contra `mapeamento_classes` no config.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar or not args.base:
        print("bases cadastradas:\n")
        for chave, (cultura, slug) in BASES.items():
            print(f"  {chave:<18} {cultura:<7} {slug}")
        print("\nUva: nao ha base consolidada de estagio de maturacao.")
        print("Procure em universe.roboflow.com (busque 'grape ripeness') ou")
        print("colete localmente. Esse e o ponto fragil do projeto.")
        return 0

    if not kaggle_disponivel():
        print("CLI do kaggle nao encontrada. Instale com: pip install kaggle")
        return 1

    return baixar(args.base)


if __name__ == "__main__":
    sys.exit(main())
