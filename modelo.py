"""
modelo.py
Arquitectura CNN para clasificación de documentos administrativo-académicos.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Umbral de confianza: scores < 0.60 indican clasificación incierta y activan revisión manual.
# Basado en Lu et al. (2022): F1-Score como métrica principal ante desbalanceo de clases.
# Categorías de documentos administrativo-académicos (Mesa de Partes Virtual — FUT, FISI-UNMSM)
CATEGORIAS = [
    "Certificados de Estudios",
    "Constancias Académicas",
    "Trámites de Convalidación",
    "Trámites de Grados y Títulos",
    "Solicitudes Administrativas Generales"
]

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


# ── Datos de entrenamiento de prueba ──────────────────────────────────────────
DATOS_ENTRENAMIENTO = [
    # Certificados de Estudios (clase 0)
    ("Solicito la emisión del certificado de estudios correspondiente a los ciclos I al VI de la carrera de Ingeniería de Sistemas, para trámite de homologación en universidad extranjera.", 0),
    ("Mediante la presente solicito se expida mi certificado de estudios completo, incluyendo todos los cursos aprobados desde el ingreso hasta el ciclo 2025-II.", 0),
    ("Solicito certificado de estudios oficial con sello de la facultad para presentar en proceso de admisión a maestría.", 0),
    ("Requiero el certificado de estudios de los últimos cuatro semestres académicos, necesario para postulación a intercambio estudiantil.", 0),
    ("Solicito la expedición de mi certificado de estudios generales, con el detalle de créditos y promedio ponderado por ciclo.", 0),
    ("Pido se me otorgue el certificado de estudios de la Escuela Profesional de Ingeniería de Sistemas para fines de trabajo en el extranjero.", 0),
    ("Solicito certificado de estudios con firma y sello de Decanato para trámite de visa de estudios.", 0),
    ("Requiero copia certificada de mi certificado de estudios, ya que el documento original fue extraviado.", 0),

    # Constancias Académicas (clase 1)
    ("Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa.", 1),
    ("Solicito constancia de notas del ciclo 2025-II para presentar ante mi centro de trabajo.", 1),
    ("Pido constancia de orden de mérito correspondiente al décimo ciclo de la carrera de Ingeniería de Sistemas.", 1),
    ("Solicito constancia de egresado, requerida para inscripción en el proceso de titulación.", 1),
    ("Solicito constancia de ingreso a la universidad, necesaria para trámite de visa estudiantil.", 1),
    ("Requiero constancia de tercio superior para postulación a beca de posgrado.", 1),
    ("Solicito constancia de no adeudo académico para completar mi expediente de graduación.", 1),
    ("Pido constancia de matrícula vigente para trámite de descuento en transporte universitario.", 1),

    # Trámites de Convalidación (clase 2)
    ("Solicito la convalidación del curso de Cálculo I llevado en la Universidad Nacional de Ingeniería, cursado en el ciclo 2023-I.", 2),
    ("Solicito convalidación de cursos aprobados en programa de intercambio en la Universidad de Chile.", 2),
    ("Pido evaluación y convalidación de la asignatura de Física General II proveniente de traslado externo.", 2),
    ("Solicito convalidación de estudios realizados en instituto superior tecnológico para continuar la carrera de Ingeniería de Sistemas.", 2),
    ("Requiero convalidación de cursos electivos cursados en universidad extranjera bajo convenio de movilidad estudiantil.", 2),
    ("Solicito la convalidación de la asignatura de Estadística I, curso aprobado en la modalidad de traslado interno.", 2),
    ("Pido revisión y convalidación de sílabos para el curso de Programación I proveniente de otra casa de estudios.", 2),
    ("Solicito convalidación de créditos obtenidos en el ciclo de nivelación de la Facultad de Ciencias Físicas.", 2),

    # Trámites de Grados y Títulos (clase 3)
    ("Solicito la inscripción de mi expediente para optar el título profesional de Ingeniero de Sistemas.", 3),
    ("Pido programación de fecha de sustentación de tesis para la obtención del título profesional.", 3),
    ("Solicito el grado académico de Bachiller en Ingeniería de Sistemas, habiendo cumplido con los requisitos establecidos.", 3),
    ("Requiero la emisión del diploma de título profesional, trámite ya aprobado por el Consejo de Facultad.", 3),
    ("Solicito duplicado de diploma de bachiller por motivo de deterioro del documento original.", 3),
    ("Pido la revisión de mi expediente de titulación por la modalidad de tesis para su aprobación final.", 3),
    ("Solicito constancia de trámite en proceso de obtención del título profesional para fines laborales.", 3),
    ("Requiero la actualización de mi expediente de grado académico de bachiller con documentos complementarios.", 3),

    # Solicitudes Administrativas Generales (clase 4)
    ("Solicito la rectificación de mis datos personales en el sistema académico debido a un error en el número de documento de identidad.", 4),
    ("Pido la emisión de un duplicado de carné universitario por pérdida del documento original.", 4),
    ("Solicito autorización para reserva de matrícula del ciclo 2026-I por motivos de salud.", 4),
    ("Requiero copia certificada de mi expediente estudiantil para trámite ante entidad externa.", 4),
    ("Solicito cambio de escuela profesional dentro de la misma facultad.", 4),
    ("Pido la devolución de tasa educativa pagada en exceso durante el proceso de matrícula.", 4),
    ("Solicito la actualización de mi correo electrónico institucional en el sistema de la universidad.", 4),
    ("Requiero autorización de retiro de curso fuera del plazo establecido por motivos justificados.", 4),
]


# ── Entrenamiento ──────────────────────────────────────────────────────────────
def entrenar_modelo():
    print("Entrenando modelo CNN...")

    textos = [d[0] for d in DATOS_ENTRENAMIENTO]
    etiquetas = [d[1] for d in DATOS_ENTRENAMIENTO]

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
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
