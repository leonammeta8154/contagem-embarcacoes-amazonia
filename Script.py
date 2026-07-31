import os
import csv

# Diretório onde este script está salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pasta_imagens = os.path.join(BASE_DIR, "imagens")
gabarito_path = os.path.join(BASE_DIR, "gabarito.csv")

extensoes = {".jpg", ".jpeg", ".png"}
imagens = sorted(
    f for f in os.listdir(pasta_imagens)
    if os.path.splitext(f)[1].lower() in extensoes
)

with open(gabarito_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["imagem", "contagem_manual"])
    for img in imagens:
        writer.writerow([img, ""])

print(f"gabarito.csv atualizado com {len(imagens)} imagens.")