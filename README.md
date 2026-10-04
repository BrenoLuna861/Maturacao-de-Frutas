# maturacao-frutos

Classificação do estágio de maturação de manga e uva a partir de imagens.

Trabalho submetido à IV MIEPE (FACAPE), novembro de 2026.

## Status

Resultados em andamento. Esta fase usa bases públicas de imagens; a coleta
própria no Vale do São Francisco, com medição de °Brix, vem depois.

Atenção na apresentação: o resumo submetido fala em aquisição sob iluminação
controlada e rotulagem por °Brix. Enquanto rodar sobre base pública, diga isso
e posicione a coleta local como etapa em curso.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

As imagens vão em `data/raw/<cultura>/<classe>/`:

```
data/raw/manga/verde/manga_001_a.jpg
data/raw/manga/verde/manga_001_b.jpg
data/raw/manga/maduro/manga_027_a.jpg
```

O sufixo depois do último `_` é a foto; o resto identifica o fruto. Fotos do
mesmo fruto formam um grupo e não se separam entre treino e teste. Se os nomes
da base não seguirem esse padrão, ajuste `regex_grupo` no config.

Nomes de classe diferentes (`ripe`, `unripe`) são convertidos pelo
`mapeamento_classes` em `configs/config.yaml`.

```bash
python -m src.data.preparar --cultura manga
python -m src.baseline --cultura manga
python -m src.treinar --cultura manga
python -m src.treinar --cultura manga --kfold
python -m src.avaliar --cultura manga
python -m src.comparar --cultura manga
```

Classificar imagens novas:

```bash
python -m src.prever --cultura manga --imagem foto.jpg
python -m src.prever --cultura manga --pasta fotos/ --csv reports/previsoes.csv
```

Para não sobrescrever a rodada anterior, nomeie o experimento:

```bash
python -m src.treinar --cultura manga --experimento lr1e4_bs32
python -m src.avaliar --cultura manga --checkpoint models/manga_lr1e4_bs32.pt
```

Cada rodada acrescenta uma linha em `reports/experimentos.csv`, e o
`comparar.py` monta a tabela e o gráfico baseline × CNN a partir dele.

Para baixar as bases públicas (precisa de credencial do Kaggle):

```bash
python scripts/baixar_dados.py --listar
python scripts/baixar_dados.py --base manga_srabon
```

Para testar o pipeline sem ter as imagens:

```bash
python scripts/gerar_dados_sinteticos.py --cultura manga --por-classe 30
python -m src.data.preparar --cultura manga
python -m src.baseline --cultura manga
```

## Notas

O split é por fruto, não por imagem. Fotos do mesmo fruto em treino e teste
inflam a acurácia; `preparar.py` aborta se detectar isso.

O baseline HSV+SVM roda em segundos e dá o número que a CNN precisa superar.

O jitter de cor está baixo de propósito. Cor é o sinal de maturação; aumento
agressivo de saturação ensina o modelo a ignorar o que importa.

`avaliar.py` salva os erros ordenados por confiança em
`reports/figures/erros_<cultura>.csv`. Erro confiante costuma ser rótulo errado
na base ou atalho aprendido (fundo, cesta, iluminação).

Imagem corrompida não derruba a execução: o treino, o baseline e a previsão
avisam no terminal e seguem. Base pública costuma ter JPEG truncado.

`reports/experimentos.csv` é versionado de propósito — é o registro de quais
números saíram de qual configuração.

## Estrutura

```
configs/config.yaml      parametros dos experimentos
src/data/preparar.py     raw/ -> splits.csv
src/data/splits.py       leitura dos splits (sem torch)
src/data/dataset.py      Dataset e transforms
src/models/construir.py  MobileNetV3
src/baseline.py          HSV + SVM
src/treinar.py           treino em duas fases, k-fold
src/avaliar.py           metricas e matriz de confusao
src/prever.py            inferencia em imagem ou pasta
src/comparar.py          tabela e grafico baseline x CNN
src/utils/registro.py    log de experimentos em CSV
scripts/                 download das bases e dados sinteticos
tests/                   testes
```

## Hardware

Testado em GTX 1660 Super (6 GB). `mobilenet_v3_small` a 224x224 com batch 32
cabe folgado. Sem GPU roda em CPU, mais devagar.
