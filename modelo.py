"""
modelo.py
Arquitectura CNN para clasificación de documentos clínicos.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Umbral de confianza: scores < 0.60 indican clasificación incierta y activan revisión manual.
# Basado en Lu et al. (2022): F1-Score como métrica principal ante desbalanceo de clases.
# Categorías de documentos clínicos
CATEGORIAS = [
    "Nota de alta médica",
    "Registro de admisión",
    "Informe de laboratorio",
    "Nota de evolución clínica",
    "Informe de imagen diagnóstica"
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
    # Notas de alta médica (clase 0)
    ("Paciente dado de alta en buen estado general. Diagnóstico: hipertensión arterial controlada. Medicación: enalapril 10mg.", 0),
    ("Alta médica voluntaria. Paciente estable, afebril. Se indica control ambulatorio en 7 días.", 0),
    ("Se procede al alta del paciente tras 3 días de hospitalización. Evolución favorable.", 0),
    ("Alta hospitalaria. Paciente con diagnóstico de neumonía resuelta. Antibioticoterapia completada.", 0),
    ("Nota de alta: paciente adulto mayor, 72 años, egresa con diagnóstico de insuficiencia cardíaca compensada.", 0),
    ("Egreso médico programado. Paciente refiere mejoría significativa. Sin fiebre en las últimas 48 horas.", 0),
    ("Alta médica. Fractura de radio distal tratada quirúrgicamente. Indicaciones postoperatorias adjuntas.", 0),

    # Registros de admisión (clase 1)
    ("Paciente ingresa por emergencia con cuadro de dolor abdominal agudo de 6 horas de evolución.", 1),
    ("Admisión hospitalaria: varón de 45 años, sin antecedentes relevantes, refiere disnea de esfuerzo.", 1),
    ("Ingreso por guardia. Paciente femenino, 32 años, con fiebre de 39°C y cefalea intensa.", 1),
    ("Registro de admisión: paciente con antecedente de diabetes mellitus tipo 2, ingresa por descompensación glucémica.", 1),
    ("Admisión de urgencia. Traumatismo craneoencefálico leve. Glasgow 14/15 al ingreso.", 1),
    ("Paciente admitido para cirugía programada de colecistectomía laparoscópica.", 1),
    ("Ingreso electivo. Paciente con diagnóstico previo de cáncer gástrico para quimioterapia.", 1),

    # Informes de laboratorio (clase 2)
    ("Hemograma completo: Hb 11.2 g/dL, leucocitos 8500/mm3, plaquetas 220000/mm3. Resultado dentro de rangos normales.", 2),
    ("Glucosa en ayunas: 126 mg/dL. Colesterol total: 210 mg/dL. Triglicéridos: 185 mg/dL.", 2),
    ("Cultivo de orina: positivo para Escherichia coli. Sensible a ciprofloxacino y nitrofurantoína.", 2),
    ("Resultado de prueba PCR COVID-19: NEGATIVO. Muestra: hisopado nasofaríngeo.", 2),
    ("Perfil hepático: TGO 45 U/L, TGP 52 U/L, bilirrubina total 1.1 mg/dL. Fosfatasa alcalina 98 U/L.", 2),
    ("Examen de orina completo: densidad 1.020, pH 6.0, proteínas negativas, glucosa negativa.", 2),
    ("Proteína C reactiva: 85 mg/L. Velocidad de sedimentación globular: 48 mm/h.", 2),

    # Notas de evolución (clase 3)
    ("Evolución: paciente refiere mejoría del dolor. Afebril. Herida quirúrgica sin signos de infección.", 3),
    ("Nota de evolución día 2: paciente tolera dieta blanda, sin náuseas ni vómitos. Funciones vitales estables.", 3),
    ("Seguimiento post-quirúrgico: herida limpia, sin secreciones. Paciente deambula sin dificultad.", 3),
    ("Nota de visita: paciente con dificultad respiratoria leve. Se ajusta oxigenoterapia a 3 litros por minuto.", 3),
    ("Evolución clínica favorable. Fiebre cedió con paracetamol. Cultivos pendientes.", 3),
    ("Control diario: presión arterial 130/85 mmHg, frecuencia cardíaca 78 lpm, saturación O2 97%.", 3),
    ("Nota de evolución: paciente consciente, orientado en tiempo y espacio. Sin déficit neurológico focal.", 3),

    # Informes de imagen (clase 4)
    ("Radiografía de tórax: cardiomegalia leve. Sin derrame pleural. Infiltrado basal derecho compatible con neumonía.", 4),
    ("Ecografía abdominal: hígado de tamaño normal, vesícula biliar con cálculo único de 12mm.", 4),
    ("Tomografía computarizada de cráneo: sin evidencia de hemorragia intracraneal. Estructuras de línea media centradas.", 4),
    ("Resonancia magnética de columna lumbar: hernia discal L4-L5 con compresión radicular leve.", 4),
    ("Ecografía obstétrica: embarazo de 20 semanas, producto único, frecuencia cardíaca fetal 148 lpm.", 4),
    ("Mamografía bilateral: BIRADS 2. Sin hallazgos sospechosos de malignidad.", 4),
    ("Tomografía de abdomen con contraste: no se evidencian masas ni colecciones. Riñones de morfología normal.", 4),
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

    categoria = CATEGORIAS[pred.item()]
    confianza = round(score.item(), 4)
    alerta = confianza < 0.60

    return {
        "categoria": categoria,
        "score_confianza": confianza,
        "alerta_revision_manual": alerta
    }
