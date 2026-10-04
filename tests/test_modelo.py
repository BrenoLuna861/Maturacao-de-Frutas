"""Testes que dependem de torch. Pulam se ele nao estiver instalado."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.data.dataset import construir_transformacoes, pesos_das_classes
from src.models.construir import congelar_backbone, construir_modelo
from src.utils.config import carregar_config


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


def test_modelo_tem_saida_do_tamanho_das_classes(cfg):
    classes = cfg["dados"]["classes"]
    modelo = construir_modelo(cfg["treino"]["arquitetura"], len(classes), False)
    entrada = torch.randn(2, 3, cfg["treino"]["img_size"], cfg["treino"]["img_size"])
    assert modelo(entrada).shape == (2, len(classes))


def test_arquitetura_invalida():
    with pytest.raises(ValueError):
        construir_modelo("resnet9000", 3, False)


def test_congelar_backbone(cfg):
    modelo = construir_modelo(cfg["treino"]["arquitetura"], 3, False)

    congelar_backbone(modelo, True)
    assert not any(p.requires_grad for p in modelo.features.parameters())
    assert all(p.requires_grad for p in modelo.classifier.parameters())

    congelar_backbone(modelo, False)
    assert all(p.requires_grad for p in modelo.features.parameters())


def test_transformacao_devolve_tensor_no_tamanho_certo(cfg):
    pytest.importorskip("PIL")
    from PIL import Image

    tamanho = cfg["treino"]["img_size"]
    imagem = Image.new("RGB", (640, 480), (120, 200, 80))

    for treino in (True, False):
        saida = construir_transformacoes(cfg, treino)(imagem)
        assert saida.shape == (3, tamanho, tamanho)


def test_pesos_das_classes_compensam_desbalanceamento():
    import pandas as pd

    # 80 da classe 0, 10 da 1, 10 da 2
    df = pd.DataFrame({"rotulo": [0] * 80 + [1] * 10 + [2] * 10})
    pesos = pesos_das_classes(df, 3)

    assert pesos[0] < pesos[1]
    assert pytest.approx(float(pesos[1]), rel=1e-3) == float(pesos[2])


def test_classe_ausente_nao_quebra():
    import pandas as pd

    df = pd.DataFrame({"rotulo": [0, 0, 1]})
    pesos = pesos_das_classes(df, 3)
    assert float(pesos[2]) == 0.0


def test_imagem_ilegivel_nao_derruba_o_dataset(tmp_path, cfg, capsys):
    import pandas as pd
    from PIL import Image

    from src.data.dataset import DatasetFrutos

    boa = tmp_path / "boa.jpg"
    Image.new("RGB", (64, 64), (10, 200, 10)).save(boa)
    ruim = tmp_path / "ruim.jpg"
    ruim.write_text("isto nao e um jpeg")

    df = pd.DataFrame({
        "caminho": [str(boa), str(ruim)],
        "rotulo": [0, 1],
    })
    ds = DatasetFrutos(df, construir_transformacoes(cfg, treino=False))

    # o item ruim cai para o proximo em vez de levantar excecao
    tensor, _ = ds[1]
    tamanho = cfg["treino"]["img_size"]
    assert tensor.shape == (3, tamanho, tamanho)
    assert "ilegivel" in capsys.readouterr().out
