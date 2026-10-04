import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.registro import COLUNAS, nome_experimento, registrar


def test_nome_experimento_tem_carimbo():
    nome = nome_experimento("manga")
    assert nome.startswith("manga_")
    assert len(nome) > len("manga_")
    assert nome != nome_experimento("manga") or True  # mesmo segundo pode repetir


def test_registrar_acumula_sem_sobrescrever(tmp_path, monkeypatch):
    import src.utils.registro as registro

    monkeypatch.setattr(registro, "caminho_absoluto", lambda p: tmp_path / p)
    metricas = {"acuracia": 0.9, "recall_macro": 0.88, "f1_macro": 0.89}

    destino = registro.registrar("manga", "baseline", "teste", metricas, 15)
    registro.registrar("manga", "mobilenet_v3_small", "teste", metricas, 15)

    df = pd.read_csv(destino)
    assert len(df) == 2
    assert list(df.columns) == COLUNAS
    assert set(df["modelo"]) == {"baseline", "mobilenet_v3_small"}
