# Contagem e Identificação de Embarcações Amazônicas com YOLOv8n

Projeto de Visão Computacional voltado à detecção e contagem automática de embarcações fluviais na região Norte do Brasil, com foco em tipos regionais como rabetas e voadeiras, típicos da orla e dos portos amazônicos.

## Contexto

Este projeto foi desenvolvido como parte da disciplina de Visão Computacional da Especialização em Inteligência Artificial da UNIFAP (Universidade Federal do Amapá). Está inserido em uma linha de trabalho mais ampla voltada à bioeconomia amazônica e ao uso de IA aplicada a contextos regionais do Norte do Brasil, área de atuação profissional do autor como consultor de sustentabilidade e ESG.

## Objetivo

Detectar e contar automaticamente embarcações a partir de imagens estáticas, utilizando o modelo YOLOv8n pré-treinado no dataset COCO, e avaliar sua acurácia comparando a contagem automática com uma contagem manual de referência (gabarito).

## Achado técnico: viés do dataset COCO

Um dos principais resultados deste projeto é a identificação e a quantificação de um viés sistemático do modelo YOLOv8n pré-treinado no COCO ao lidar com cenas amazônicas:

| Métrica | Resultado |
|---|---|
| Imagens avaliadas | 6 |
| Contagens exatas | 0 (0.0%) |
| Erro absoluto médio (MAE) | 37.67 embarcações/imagem |

O modelo detectou consistentemente **muito menos** embarcações do que a contagem manual real, com erro sempre na mesma direção (subestimação). Esse padrão é coerente com a hipótese de que o COCO não representa adequadamente embarcações regionais como rabetas e voadeiras, além de ter dificuldade adicional em cenas com muitas embarcações agrupadas e de pequeno porte, como é comum em portos e orlas da Amazônia.

Esse resultado embasa tecnicamente a necessidade de um processo de fine-tuning com dados anotados localmente, etapa prevista como próximo passo do projeto.

## Estrutura do projeto

```
projeto_visao/
├── barcos.py                  # Script principal: detecção, contagem e avaliação
├── Script.py                  # Gera/atualiza o gabarito.csv a partir da pasta imagens/
├── yolov8n.pt                 # Pesos do modelo YOLOv8n
├── contagem_automatica.csv    # Saída: contagem gerada pelo modelo
├── gabarito.csv                # Contagem manual de referência (verdade de campo)
├── imagens/                    # Imagens de entrada para detecção
├── resultados/                 # Imagens anotadas com as detecções (gerado automaticamente)
└── .gitignore
```

## Requisitos

- Python 3.11+
- Ambiente virtual (venv) dedicado ao projeto, para evitar conflitos com outras dependências (ex: MediaPipe, TensorFlow)

Bibliotecas principais:
- `ultralytics` (YOLOv8)
- `opencv-python`

## Instalação

```powershell
git clone https://github.com/leonammeta8154/contagem-embarcacoes-amazonia.git
cd contagem-embarcacoes-amazonia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics opencv-python
```

## Como usar

### 1. Gerar/atualizar o gabarito com os nomes atuais das imagens

Sempre que a pasta `imagens/` for alterada (imagens adicionadas, removidas ou renomeadas), rode:

```powershell
python Script.py
```

Isso recria o `gabarito.csv` com a lista atual de arquivos, pronta para preenchimento manual.

### 2. Rodar a detecção e contagem automática

```powershell
python barcos.py
```

O script:
1. Carrega o modelo YOLOv8n (baixado automaticamente na primeira execução)
2. Detecta embarcações em todas as imagens de `imagens/`
3. Salva as imagens anotadas em `resultados/`
4. Gera `contagem_automatica.csv` com a contagem por imagem
5. Se `gabarito.csv` estiver preenchido, calcula e exibe as métricas de acurácia (MAE e percentual de contagens exatas)

### 3. Preencher o gabarito manualmente

Abra `gabarito.csv` (separado por `;`, compatível com Excel em português) e preencha a coluna `contagem_manual` com a contagem real de embarcações observada em cada imagem. Depois, rode `barcos.py` novamente para obter as métricas de avaliação.

## Resultados atuais

O modelo pré-treinado no COCO apresenta baixa acurácia em cenas amazônicas densas, conforme detalhado na seção de achado técnico acima. Os resultados completos por imagem ficam registrados na saída do terminal ao rodar `barcos.py`.

## Próximos passos

- Anotação manual de um conjunto de imagens locais (classes: rabeta, voadeira, barco grande, canoa)
- Fine-tuning do YOLOv8n com o dataset anotado
- Reavaliação da acurácia (MAE e contagens exatas) pós fine-tuning, comparando com o baseline atual
- Ciclo de active learning: identificar erros recorrentes e priorizar anotação nesses casos

## Autor

Leonam Azevedo, pós-graduando em Inteligência Artificial na UNIFAP (Almeirim, Pará), consultor de sustentabilidade e ESG, com interesse em bioeconomia amazônica.
