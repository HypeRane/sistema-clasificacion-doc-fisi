"""
modelo.py
Arquitectura CNN para clasificación de Resoluciones Decanales de la FISI-UNMSM.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json

# Umbral de confianza: scores < 0.60 indican clasificación incierta y activan revisión manual.
# Basado en Lu et al. (2022): F1-Score como métrica principal ante desbalanceo de clases.
# Categorías de Resoluciones Decanales (datos públicos del portal de transparencia FISI-UNMSM).
# Piloto de validación de concepto mientras Registros Académicos completa la digitalización
# de los documentos del FUT (ver Capítulo IV — Dataset de Entrenamiento).
CATEGORIAS = [
    "Grados Académicos y Títulos Profesionales",
    "Gestión Académica, Económica y Normativa General",
    "Trámites de Matrícula",
    "Rectificación y Protección de Datos Personales",
    "Gestión de Personal Docente y Administrativo",
]

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dataset", "resoluciones_decanales", "dataset.json")

# ── Arquitectura CNN ───────────────────────────────────────────────────────────
class CNN_Clasificador(nn.Module):
    def __init__(self, vocab_size, num_clases=5, embed_dim=64):
        super(CNN_Clasificador, self).__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_clases)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.embedding(x))   # (batch, embed_dim)
        x = x.unsqueeze(1)                 # (batch, 1, embed_dim)
        x = self.relu(self.conv1(x))       # (batch, 64, embed_dim)
        x = self.relu(self.conv2(x))       # (batch, 128, embed_dim)
        x = self.pool(x).squeeze(2)        # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)                     # (batch, num_clases)
        return x


# ── Datos de entrenamiento: Resoluciones Decanales reales (FISI-UNMSM) ─────────
def cargar_datos_entrenamiento():
    """
    Carga el corpus real de entrenamiento desde dataset/resoluciones_decanales/dataset.json
    (197 Resoluciones Decanales descargadas del portal de transparencia FISI-UNMSM,
    etiquetadas por categoría). Cada elemento es (texto, categoria_id).
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset en {DATASET_PATH}. "
            "Se requiere dataset/resoluciones_decanales/dataset.json para entrenar el modelo."
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        registros = json.load(f)
    return [(r["texto"], CATEGORIAS.index(r["categoria"])) for r in registros]


# No se carga a nivel de módulo: si ya existen modelo_cnn.pt/vectorizer.pkl,
# cargar_modelo() los usa directamente y nunca necesita el dataset crudo
# (que no viaja con la imagen Docker ni, opcionalmente, con el repositorio).


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar_modelo():
    print("Entrenando modelo CNN...")

    datos_entrenamiento = cargar_datos_entrenamiento()
    textos = [d[0] for d in datos_entrenamiento]
    etiquetas = [d[1] for d in datos_entrenamiento]

    # Vectorización TF-IDF. max_df descarta el vocabulario legal repetido en casi
    # todas las resoluciones (VISTO, CONSIDERANDO, Ley Universitaria N.° 30220...),
    # que de lo contrario dominaría por frecuencia sin aportar señal de categoría;
    # min_df descarta términos que aparecen una sola vez (a menudo nombres propios).
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X = vectorizer.fit_transform(textos).toarray()
    y = torch.tensor(etiquetas, dtype=torch.long)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    vocab_size = X.shape[1]
    modelo = CNN_Clasificador(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    modelo.train()
    for epoch in range(150):
        optimizer.zero_grad()
        output = modelo(X_tensor)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Época {epoch+1}/150 — Loss: {loss.item():.4f}")

    # Guardar modelo y vectorizador
    torch.save(modelo.state_dict(), "modelo_cnn.pt")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Modelo entrenado y guardado correctamente.")
    return modelo, vectorizer


# ── Carga del modelo ───────────────────────────────────────────────────────────
def cargar_modelo():
    if not os.path.exists("modelo_cnn.pt") or not os.path.exists("vectorizer.pkl"):
        return entrenar_modelo()

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    vocab_size = len(vectorizer.vocabulary_)
    modelo = CNN_Clasificador(vocab_size=vocab_size)
    modelo.load_state_dict(torch.load("modelo_cnn.pt", weights_only=True))
    modelo.eval()
    print("Modelo cargado desde archivo.")
    return modelo, vectorizer


# ── Inferencia ─────────────────────────────────────────────────────────────────
def clasificar(texto: str, modelo, vectorizer) -> dict:
    modelo.eval()
    with torch.no_grad():
        X = vectorizer.transform([texto]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        output = modelo(X_tensor)
        probs = torch.softmax(output, dim=1)
        score, pred = torch.max(probs, dim=1)

    categoria_id = pred.item()
    categoria = CATEGORIAS[categoria_id]
    confianza = round(score.item(), 4)
    alerta = confianza < 0.60

    # Explicabilidad: términos con mayor peso TF-IDF presentes en el texto de entrada.
    vector_fila = X[0]
    nombres_terminos = vectorizer.get_feature_names_out()
    indices_top = np.argsort(vector_fila)[::-1]
    terminos_clave = [nombres_terminos[i] for i in indices_top if vector_fila[i] > 0][:5]

    return {
        "categoria": categoria,
        "categoria_id": categoria_id,
        "score_confianza": confianza,
        "alerta_revision_manual": alerta,
        "terminos_clave": terminos_clave
    }
