import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preparar import _conferir_vazamento, _extrair_grupo, _normalizar_classe
from src.utils.config import carregar_config
from src.utils.metricas import calcular_metricas

CLASSES = ["verde", "de_vez", "maduro"]


def test_split_soma_um():
    s = carregar_config()["dados"]["split"]
    assert abs(s["treino"] + s["validacao"] + s["teste"] - 1.0) < 1e-6


def test_mapeamento_aponta_para_classe_valida():
    cfg = carregar_config()
    classes = set(cfg["dados"]["classes"])
    for origem, destino in (cfg["dados"].get("mapeamento_classes") or {}).items():
        assert destino in classes, f"{origem} -> {destino} nao existe"


def test_normalizar_classe():
    mapa = {"unripe": "verde", "ripe": "maduro"}
    assert _normalizar_classe("Verde", mapa, CLASSES) == "verde"
    assert _normalizar_classe("UNRIPE", mapa, CLASSES) == "verde"
    assert _normalizar_classe("de-vez", mapa, CLASSES) == "de_vez"
    assert _normalizar_classe("banana", mapa, CLASSES) is None


def test_extrair_grupo():
    regex = carregar_config()["dados"]["regex_grupo"]
    assert _extrair_grupo("manga_017_a.jpg", regex) == "manga_017"
    assert _extrair_grupo("manga_017_b.jpg", regex) == "manga_017"


def test_vazamento():
    ok = pd.DataFrame({"grupo": ["g1", "g1", "g2"], "split": ["treino", "treino", "teste"]})
    _conferir_vazamento(ok)

    with pytest.raises(AssertionError):
        _conferir_vazamento(
            pd.DataFrame({"grupo": ["g1", "g1"], "split": ["treino", "teste"]})
        )


def test_metricas():
    m = calcular_metricas([0, 1, 2], [0, 1, 2], CLASSES)
    assert m["acuracia"] == 1.0
    assert m["f1_macro"] == 1.0
