import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.splits import carregar_splits
from src.utils.config import caminho_absoluto

MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


class DatasetFrutos(Dataset):
    def __init__(self, df, transformacao):
        self.df = df.reset_index(drop=True)
        self.transformacao = transformacao

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        linha = self.df.iloc[i]
        imagem = Image.open(caminho_absoluto(linha["caminho"])).convert("RGB")
        return self.transformacao(imagem), int(linha["rotulo"])


def construir_transformacoes(cfg, treino):
    tamanho = cfg["treino"]["img_size"]

    if not treino:
        return transforms.Compose([
            transforms.Resize((tamanho, tamanho)),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIO),
        ])

    aug = cfg["aumento_dados"]
    etapas = [transforms.RandomResizedCrop(tamanho, scale=(0.7, 1.0))]

    if aug.get("flip_horizontal"):
        etapas.append(transforms.RandomHorizontalFlip())
    if aug.get("rotacao_graus"):
        etapas.append(transforms.RandomRotation(aug["rotacao_graus"]))

    # cuidado com valores altos aqui: a cor e o proprio sinal de maturacao
    if any(aug.get(k) for k in ("jitter_brilho", "jitter_contraste", "jitter_saturacao")):
        etapas.append(transforms.ColorJitter(
            brightness=aug.get("jitter_brilho", 0),
            contrast=aug.get("jitter_contraste", 0),
            saturation=aug.get("jitter_saturacao", 0),
        ))

    etapas += [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIO)]
    return transforms.Compose(etapas)


def criar_dataloaders(cultura, cfg):
    df = carregar_splits(cultura, cfg)
    loaders = {}

    for split in ("treino", "validacao", "teste"):
        parte = df[df["split"] == split]
        if parte.empty:
            continue
        eh_treino = split == "treino"
        loaders[split] = DataLoader(
            DatasetFrutos(parte, construir_transformacoes(cfg, eh_treino)),
            batch_size=cfg["treino"]["batch_size"],
            shuffle=eh_treino,
            num_workers=cfg["treino"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def pesos_das_classes(df, n_classes):
    """Peso inverso a frequencia, para base desbalanceada."""
    contagem = df["rotulo"].value_counts().reindex(range(n_classes), fill_value=0)
    total = contagem.sum()
    pesos = [(total / (n_classes * c)) if c > 0 else 0.0 for c in contagem]
    return torch.tensor(pesos, dtype=torch.float32)
