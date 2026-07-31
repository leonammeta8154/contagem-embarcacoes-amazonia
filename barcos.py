"""
Deteccao e contagem de embarcacoes com YOLO pre-treinado
Disciplina: Visao Computacional
Projeto: contagem de embarcacoes na orla e no porto (regiao Norte / Amapa)

O que este script faz:
    1. Carrega um modelo YOLO pre-treinado (classe "boat" do dataset COCO)
    2. Percorre todas as imagens da pasta ./imagens
    3. Detecta e conta as embarcacoes em cada uma
    4. Salva as imagens anotadas em ./resultados
    5. Gera um CSV com a contagem automatica de cada imagem
    6. Se existir um gabarito manual (gabarito.csv), compara e reporta a acuracia

Uso:
    .\.venv\Scripts\python.exe barcos.py

Fluxo de trabalho sugerido:
    - deposite fotos e frames na pasta ./imagens
    - rode o script uma vez para gerar a contagem automatica
    - preencha o gabarito.csv com a contagem manual (verdade de referencia)
    - rode novamente para obter as metricas de acuracia
"""

import os
import csv
import sys

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "[ERRO] Biblioteca 'ultralytics' nao encontrada.\n"
        "Instale no ambiente do projeto com:\n"
        "    .\\.venv\\Scripts\\python.exe -m pip install ultralytics"
    )

# ---------------------------------------------------------------
# Configuracao de caminhos (relativos a este arquivo)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, "imagens")
PASTA_RESULTADOS = os.path.join(BASE_DIR, "resultados")
CSV_CONTAGEM = os.path.join(BASE_DIR, "contagem_automatica.csv")
CSV_GABARITO = os.path.join(BASE_DIR, "gabarito.csv")

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# ---------------------------------------------------------------
# Parametros do detector
# ---------------------------------------------------------------
# yolov8n = versao "nano", a mais leve, roda bem em CPU. Se a maquina
# aguentar e a precisao for baixa, troque por yolov8s ou yolov8m.
MODELO = "yolov8n.pt"

# Limiar de confianca: deteccoes abaixo disso sao descartadas. Comece
# em 0.25 (padrao) e ajuste na fase de analise. Valores altos reduzem
# falsos positivos mas aumentam falsos negativos.
CONFIANCA = 0.25

# No dataset COCO, "boat" e a classe de indice 8. Filtramos apenas ela.
CLASSE_BARCO = 8

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Delimitador usado nos CSVs gerados/lidos. Excel em portugues-BR espera ";"
# para separar colunas automaticamente ao abrir o arquivo com duplo-clique.
DELIMITADOR_CSV = ";"

# ---------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------
print("=" * 60)
print("Deteccao e contagem de embarcacoes - YOLO")
print("=" * 60)
print(f"[INFO] Carregando modelo {MODELO}...")
print("[INFO] Na primeira execucao, o modelo sera baixado automaticamente.")

modelo = YOLO(MODELO)

# ---------------------------------------------------------------
# Coleta das imagens
# ---------------------------------------------------------------
if not os.path.isdir(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    sys.exit(
        f"[AVISO] A pasta de imagens foi criada agora: {PASTA_IMAGENS}\n"
        "Deposite fotos e frames nela e rode o script novamente."
    )

arquivos = sorted(
    f for f in os.listdir(PASTA_IMAGENS)
    if f.lower().endswith(EXTENSOES)
)

if not arquivos:
    sys.exit(
        f"[AVISO] Nenhuma imagem encontrada em {PASTA_IMAGENS}.\n"
        "Formatos aceitos: " + ", ".join(EXTENSOES)
    )

print(f"[INFO] {len(arquivos)} imagem(ns) encontrada(s).\n")

# ---------------------------------------------------------------
# Deteccao imagem a imagem
# ---------------------------------------------------------------
resultados = []

print(f"{'Imagem':<32} {'Embarcacoes':>12}")
print("-" * 60)

for nome in arquivos:
    caminho = os.path.join(PASTA_IMAGENS, nome)

    # cv2 nao le caminhos com acento de forma confiavel no Windows;
    # como a pasta do projeto nao tem acento, a leitura direta e segura
    predicoes = modelo(caminho, conf=CONFIANCA, verbose=False)

    contagem = 0
    imagem = cv2.imread(caminho)

    for pred in predicoes:
        for caixa in pred.boxes:
            if int(caixa.cls[0]) == CLASSE_BARCO:
                contagem += 1
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                conf = float(caixa.conf[0])
                cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imagem, f"barco {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Carimba a contagem total no canto da imagem
    cv2.putText(imagem, f"Total: {contagem}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    destino = os.path.join(PASTA_RESULTADOS, nome)
    cv2.imwrite(destino, imagem)

    resultados.append({"imagem": nome, "contagem_auto": contagem})
    print(f"{nome:<32} {contagem:>12}")

print("-" * 60)
total = sum(r["contagem_auto"] for r in resultados)
print(f"{'TOTAL GERAL':<32} {total:>12}\n")

# ---------------------------------------------------------------
# Gravacao da contagem automatica
# ---------------------------------------------------------------
with open(CSV_CONTAGEM, "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=["imagem", "contagem_auto"], delimiter=DELIMITADOR_CSV)
    escritor.writeheader()
    escritor.writerows(resultados)

print(f"[INFO] Contagem salva em: {CSV_CONTAGEM}")
print(f"[INFO] Imagens anotadas em: {PASTA_RESULTADOS}")

# Cria um modelo de gabarito se ainda nao existir, para o aluno preencher
if not os.path.exists(CSV_GABARITO):
    with open(CSV_GABARITO, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["imagem", "contagem_manual"], delimiter=DELIMITADOR_CSV)
        escritor.writeheader()
        for r in resultados:
            escritor.writerow({"imagem": r["imagem"], "contagem_manual": ""})
    print(f"[INFO] Modelo de gabarito criado: {CSV_GABARITO}")
    print("       Preencha a coluna 'contagem_manual' e rode novamente.")

# ---------------------------------------------------------------
# Avaliacao contra o gabarito (se preenchido)
# ---------------------------------------------------------------
def carregar_gabarito(caminho):
    """Le o gabarito e retorna dict {imagem: contagem_manual} apenas
    para linhas preenchidas com numero valido."""
    gabarito = {}
    if not os.path.exists(caminho):
        return gabarito
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f, delimiter=DELIMITADOR_CSV):
            valor = (linha.get("contagem_manual") or "").strip()
            if valor.isdigit():
                gabarito[linha["imagem"]] = int(valor)
    return gabarito


gabarito = carregar_gabarito(CSV_GABARITO)

if gabarito:
    print("\n" + "=" * 60)
    print("AVALIACAO CONTRA O GABARITO MANUAL")
    print("=" * 60)
    print(f"{'Imagem':<28} {'Auto':>6} {'Manual':>8} {'Erro':>6}")
    print("-" * 60)

    erro_absoluto_total = 0
    exatas = 0
    avaliadas = 0

    for r in resultados:
        nome = r["imagem"]
        if nome not in gabarito:
            continue
        auto = r["contagem_auto"]
        manual = gabarito[nome]
        erro = auto - manual
        erro_absoluto_total += abs(erro)
        if erro == 0:
            exatas += 1
        avaliadas += 1
        print(f"{nome:<28} {auto:>6} {manual:>8} {erro:>+6}")

    print("-" * 60)
    if avaliadas:
        mae = erro_absoluto_total / avaliadas
        pct_exatas = 100.0 * exatas / avaliadas
        print(f"Imagens avaliadas          : {avaliadas}")
        print(f"Contagens exatas           : {exatas} ({pct_exatas:.1f}%)")
        print(f"Erro absoluto medio (MAE)  : {mae:.2f} embarcacoes/imagem")
    print("=" * 60)
else:
    print("\n[INFO] Gabarito ainda nao preenchido. Preencha "
          "gabarito.csv para ver as metricas de acuracia.")
