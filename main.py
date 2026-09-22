"""
main.py
API REST del Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos.
Universidad Nacional Mayor de San Marcos — FISI
Autor: Ortiz Herrera, Fabrizio Peter
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from modelo import cargar_modelo, clasificar, CATEGORIAS
import uvicorn
import time
import os
import json
import csv
import io
import sqlite3

app = FastAPI(
    title="Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos",
    description="""
## Sistema CNN para la Mesa de Partes Virtual — FISI-UNMSM

Este sistema utiliza una **Red Neuronal Convolucional (CNN)** entrenada sobre
solicitudes del Formato Único de Trámite (FUT) en español para clasificar
automáticamente documentos administrativo-académicos en cinco categorías:

- Certificados de Estudios
- Constancias Académicas
- Trámites de Convalidación
- Trámites de Grados y Títulos
- Solicitudes Administrativas Generales

**Universidad Nacional Mayor de San Marcos — FISI | Tesis de pregrado 2025**
    """,
    version="1.2.0",
    contact={"name": "Ortiz Herrera, Fabrizio Peter", "email": "fabrizio.ortiz@unmsm.edu.pe"}
)

app.mount("/static", StaticFiles(directory="static"), name="static")

print("Iniciando sistema...")
modelo_cnn, vectorizer = cargar_modelo()
print("Sistema listo.")

# ── Persistencia de categorías (renombrables sin reentrenar el modelo) ─────────
CATEGORIAS_PATH = "categorias.json"

def cargar_categorias_actuales() -> list[str]:
    if os.path.exists(CATEGORIAS_PATH):
        try:
            with open(CATEGORIAS_PATH, "r", encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, list) and len(datos) == len(CATEGORIAS) and all(isinstance(c, str) for c in datos):
                return datos
        except (json.JSONDecodeError, OSError):
            pass
    return list(CATEGORIAS)

def guardar_categorias(categorias: list[str]) -> None:
    with open(CATEGORIAS_PATH, "w", encoding="utf-8") as f:
        json.dump(categorias, f, ensure_ascii=False, indent=2)

categorias_actuales = cargar_categorias_actuales()

def nombre_categoria(idx: int) -> str:
    if 0 <= idx < len(categorias_actuales):
        return categorias_actuales[idx]
    return f"Categoría {idx}"


# ── Persistencia del historial (SQLite) ─────────────────────────────────────────
DB_PATH = "historial.db"

def obtener_conexion() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def inicializar_db() -> None:
    conn = obtener_conexion()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto_fragmento TEXT NOT NULL,
            categoria_id INTEGER NOT NULL,
            score_confianza REAL NOT NULL,
            alerta_revision_manual INTEGER NOT NULL,
            tiempo_inferencia_ms REAL NOT NULL,
            terminos_clave TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

inicializar_db()


class DocumentoEntrada(BaseModel):
    texto: str = Field(..., min_length=10,
        description="Texto de la solicitud o documento administrativo-académico a clasificar",
        json_schema_extra={"example": "Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa."})

class ResultadoClasificacion(BaseModel):
    categoria: str
    score_confianza: float
    alerta_revision_manual: bool
    terminos_clave: list[str]
    tiempo_inferencia_ms: float
    timestamp: str

class RegistroHistorial(BaseModel):
    id: int
    texto_fragmento: str
    categoria: str
    score_confianza: float
    alerta_revision_manual: bool
    tiempo_inferencia_ms: float
    timestamp: str

class MetricasSistema(BaseModel):
    total_documentos_procesados: int
    documentos_con_alerta: int
    porcentaje_alertas: float
    distribucion_por_categoria: dict
    tiempo_promedio_inferencia_ms: float
    categorias_disponibles: list

class AlertaDetalle(BaseModel):
    id: int
    texto_fragmento: str
    score_confianza: float
    categoria_asignada: str
    timestamp: str

class CategoriasUpdate(BaseModel):
    categorias: list[str] = Field(...,
        description="Nuevos nombres para las 5 categorías, en el mismo orden de clases del modelo entrenado (no se pueden agregar ni quitar sin reentrenar).")


@app.get("/", response_class=HTMLResponse, tags=["Interfaz"], include_in_schema=False)
def interfaz_web():
    """Interfaz web visual del sistema de clasificación."""
    return HTML_INTERFAZ


@app.get("/estado", tags=["Sistema"])
def estado():
    """Verifica que el sistema está activo."""
    return {"sistema": "Clasificación Automatizada de Documentos Administrativo-Académicos",
            "estado": "activo", "version": "1.2.0", "documentacion": "/docs"}


@app.post("/clasificar", response_model=ResultadoClasificacion, tags=["Clasificación"])
def clasificar_documento(documento: DocumentoEntrada):
    """
    Clasifica un documento administrativo-académico en una de las cinco categorías definidas.

    - Recibe el **texto** de la solicitud o documento
    - Devuelve la **categoría asignada**, el **score de confianza** (0-1) y los **términos clave**
      (TF-IDF) que más influyeron en la predicción
    - Incluye el **tiempo de inferencia** en milisegundos
    - Si el score es menor a 0.60, genera una **alerta de revisión manual**
    """
    try:
        inicio = time.perf_counter()
        resultado = clasificar(documento.texto, modelo_cnn, vectorizer)
        tiempo_ms = round((time.perf_counter() - inicio) * 1000, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        categoria_id = resultado["categoria_id"]
        categoria_nombre = nombre_categoria(categoria_id)
        texto_fragmento = documento.texto[:80] + "..." if len(documento.texto) > 80 else documento.texto

        conn = obtener_conexion()
        conn.execute(
            "INSERT INTO documentos (texto_fragmento, categoria_id, score_confianza, alerta_revision_manual, "
            "tiempo_inferencia_ms, terminos_clave, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (texto_fragmento, categoria_id, resultado["score_confianza"], int(resultado["alerta_revision_manual"]),
             tiempo_ms, ",".join(resultado["terminos_clave"]), timestamp)
        )
        conn.commit()
        conn.close()

        return ResultadoClasificacion(
            categoria=categoria_nombre,
            score_confianza=resultado["score_confianza"],
            alerta_revision_manual=resultado["alerta_revision_manual"],
            terminos_clave=resultado["terminos_clave"],
            tiempo_inferencia_ms=tiempo_ms,
            timestamp=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")


@app.get("/historial", response_model=list[RegistroHistorial], tags=["Historial"])
def obtener_historial(categoria: Optional[str] = None, solo_alertas: Optional[bool] = False):
    """Devuelve el historial de documentos procesados, más reciente primero."""
    conn = obtener_conexion()
    filas = conn.execute("SELECT * FROM documentos ORDER BY id DESC").fetchall()
    conn.close()

    resultado = []
    for f in filas:
        nombre_cat = nombre_categoria(f["categoria_id"])
        if categoria and categoria.lower() not in nombre_cat.lower():
            continue
        if solo_alertas and not f["alerta_revision_manual"]:
            continue
        resultado.append(RegistroHistorial(
            id=f["id"],
            texto_fragmento=f["texto_fragmento"],
            categoria=nombre_cat,
            score_confianza=f["score_confianza"],
            alerta_revision_manual=bool(f["alerta_revision_manual"]),
            tiempo_inferencia_ms=f["tiempo_inferencia_ms"],
            timestamp=f["timestamp"]
        ))
    return resultado


@app.get("/historial/exportar", tags=["Historial"])
def exportar_historial_csv():
    """Exporta el historial completo de clasificaciones como archivo CSV."""
    conn = obtener_conexion()
    filas = conn.execute("SELECT * FROM documentos ORDER BY id").fetchall()
    conn.close()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "texto_fragmento", "categoria", "score_confianza",
                      "alerta_revision_manual", "tiempo_inferencia_ms", "terminos_clave", "timestamp"])
    for f in filas:
        writer.writerow([
            f["id"], f["texto_fragmento"], nombre_categoria(f["categoria_id"]), f["score_confianza"],
            bool(f["alerta_revision_manual"]), f["tiempo_inferencia_ms"], f["terminos_clave"], f["timestamp"]
        ])
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=historial_clasificaciones.csv"}
    )


@app.get("/metricas", response_model=MetricasSistema, tags=["Monitoreo"])
def obtener_metricas():
    """Devuelve métricas de rendimiento del sistema."""
    conn = obtener_conexion()
    filas = conn.execute("SELECT categoria_id, alerta_revision_manual, tiempo_inferencia_ms FROM documentos").fetchall()
    conn.close()

    total = len(filas)
    alertas = sum(1 for f in filas if f["alerta_revision_manual"])
    porcentaje = round((alertas / total * 100), 2) if total > 0 else 0.0

    distribucion = {nombre: 0 for nombre in categorias_actuales}
    tiempos = []
    for f in filas:
        nombre_cat = nombre_categoria(f["categoria_id"])
        distribucion[nombre_cat] = distribucion.get(nombre_cat, 0) + 1
        tiempos.append(f["tiempo_inferencia_ms"])
    tiempo_prom = round(sum(tiempos) / len(tiempos), 2) if tiempos else 0.0

    return MetricasSistema(
        total_documentos_procesados=total,
        documentos_con_alerta=alertas,
        porcentaje_alertas=porcentaje,
        distribucion_por_categoria=distribucion,
        tiempo_promedio_inferencia_ms=tiempo_prom,
        categorias_disponibles=categorias_actuales
    )


@app.get("/alertas", response_model=list[AlertaDetalle], tags=["Monitoreo"])
def obtener_alertas():
    """Devuelve la lista de documentos que requieren revisión manual."""
    conn = obtener_conexion()
    filas = conn.execute("SELECT * FROM documentos WHERE alerta_revision_manual = 1 ORDER BY id DESC").fetchall()
    conn.close()
    return [AlertaDetalle(id=f["id"], texto_fragmento=f["texto_fragmento"],
            score_confianza=f["score_confianza"], categoria_asignada=nombre_categoria(f["categoria_id"]),
            timestamp=f["timestamp"]) for f in filas]


@app.put("/categorias", tags=["Administración"])
def actualizar_categorias(payload: Optional[CategoriasUpdate] = None):
    """
    Sin cuerpo: devuelve las categorías actualmente configuradas.
    Con cuerpo `{"categorias": [...]}`: renombra las 5 categorías (el número de clases
    está fijado por la arquitectura del modelo entrenado; no se pueden agregar ni quitar
    sin reentrenar la CNN).
    """
    global categorias_actuales
    if payload is None:
        return {"categorias": categorias_actuales, "total": len(categorias_actuales),
                "mensaje": "Categorías activas en el sistema."}

    if len(payload.categorias) != len(CATEGORIAS):
        raise HTTPException(status_code=400,
            detail=f"Se requieren exactamente {len(CATEGORIAS)} nombres (uno por clase del modelo entrenado).")

    nombres = [c.strip() for c in payload.categorias]
    if any(not c for c in nombres):
        raise HTTPException(status_code=400, detail="Los nombres de categoría no pueden estar vacíos.")

    categorias_actuales = nombres
    guardar_categorias(categorias_actuales)
    return {"categorias": categorias_actuales, "total": len(categorias_actuales),
            "mensaje": "Categorías actualizadas correctamente."}


# ══════════════════════════════════════════════════════════════════════════════
# HTML de la interfaz
# ══════════════════════════════════════════════════════════════════════════════
HTML_INTERFAZ = r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Clasificación de Documentos Administrativo-Académicos — UNMSM</title>
    <style>
        :root {
            --guinda: #731E19;
            --guinda-oscuro: #4E1310;
            --guinda-claro: #A54A3F;
            --dorado: #B8902E;
            --dorado-claro: #D9B65C;
            --crema: #FBF7F2;
            --gris-fondo: #F4F6F9;
            --gris-borde: #E0E6ED;
            --texto: #333;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, var(--guinda-oscuro) 0%, var(--guinda) 100%); min-height: 100vh; padding: 20px; color: var(--texto); }
        .container { max-width: 1050px; margin: 0 auto; }
        .cinta-dorada { height: 5px; max-width: 1050px; margin: 0 auto; background: linear-gradient(90deg, var(--dorado-claro), var(--dorado), var(--dorado-claro)); border-radius: 4px; }
        .header { text-align: center; background: var(--crema); border-radius: 16px; padding: 26px 20px 20px; margin: 20px 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .logo-unmsm { max-width: 320px; width: 80%; height: auto; margin: 0 auto 14px; display: block; }
        .header h1 { font-size: 1.5rem; margin-bottom: 8px; font-weight: 700; color: var(--guinda); }
        .header p { font-size: 0.95rem; opacity: 0.85; color: #555; }
        .badge { display: inline-block; background: var(--guinda); border: 1px solid var(--guinda); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin-top: 10px; color: var(--dorado-claro); }
        .card { background: white; border-radius: 16px; padding: 26px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px; }
        .card h2 { font-size: 1.15rem; color: var(--guinda); margin-bottom: 16px; border-left: 4px solid var(--dorado); padding-left: 10px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        textarea { width: 100%; min-height: 110px; padding: 14px; border: 2px solid var(--gris-borde); border-radius: 10px; font-size: 0.95rem; font-family: inherit; resize: vertical; transition: border 0.2s; }
        textarea:focus { outline: none; border-color: var(--guinda-claro); }
        .ejemplos-grupo { margin: 14px 0; }
        .titulo-grupo { font-size: 0.78rem; color: #888; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .ejemplos { display: flex; flex-wrap: wrap; gap: 8px; }
        .ejemplo-btn { background: #F5E9E7; border: 1px solid #E0C6C2; color: var(--guinda); padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
        .ejemplo-btn:hover { background: var(--guinda); color: white; }
        .btn-aleatorio { background: var(--dorado); border-color: var(--dorado); color: white; font-weight: 600; }
        .btn-aleatorio:hover { background: #96731F; }
        .btn-clasificar { width: 100%; background: linear-gradient(135deg, var(--guinda-oscuro), var(--guinda)); color: white; border: none; padding: 14px; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 14px; transition: transform 0.1s; }
        .btn-clasificar:hover { transform: translateY(-2px); }
        .btn-secundario { background: var(--dorado); color: white; border: none; padding: 10px 18px; border-radius: 8px; font-size: 0.88rem; font-weight: 600; cursor: pointer; margin-top: 12px; }
        .btn-secundario:hover { background: #96731F; }
        .btn-exportar { display: inline-block; margin-top: 14px; background: #1B7A72; color: white; padding: 9px 16px; border-radius: 8px; text-decoration: none; font-size: 0.85rem; font-weight: 600; }
        .btn-exportar:hover { background: #145E58; }
        .resultado { margin-top: 20px; padding: 20px; border-radius: 12px; display: none; animation: fadeIn 0.4s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; } }
        .resultado.exito { background: #EAF7EC; border: 2px solid #27AE60; }
        .resultado.dudoso { background: #FEF9E7; border: 2px solid #F39C12; }
        .resultado.alerta { background: #FDEDEC; border: 2px solid #E74C3C; }
        .resultado h3 { font-size: 1.3rem; margin-bottom: 6px; }
        .resultado.exito h3 { color: #1D6A35; }
        .resultado.dudoso h3 { color: #935A0A; }
        .resultado.alerta h3 { color: #922B21; }
        .tiempo-info { font-size: 0.82rem; color: #666; margin-bottom: 12px; }
        .score-bar { background: var(--gris-borde); border-radius: 20px; height: 26px; margin: 10px 0; overflow: hidden; }
        .score-fill { height: 100%; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-size: 0.82rem; font-weight: 600; transition: width 0.8s ease; width: 0%; }
        .score-fill.alto { background: linear-gradient(90deg, #27AE60, #2ECC71); }
        .score-fill.medio { background: linear-gradient(90deg, #F39C12, #F1C40F); }
        .score-fill.bajo { background: linear-gradient(90deg, #E74C3C, #EC7063); }
        .chip { display: inline-block; background: #F5E9E7; color: var(--guinda); padding: 4px 10px; border-radius: 14px; font-size: 0.78rem; margin: 0 6px 6px 0; border: 1px solid #E0C6C2; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
        .stat-box { background: var(--gris-fondo); border-radius: 10px; padding: 16px; text-align: center; border: 1px solid var(--gris-borde); }
        .stat-box .num { font-size: 1.7rem; font-weight: 700; color: var(--guinda); }
        .stat-box .label { font-size: 0.78rem; color: #666; margin-top: 4px; }
        .chart-wrap { position: relative; height: 260px; }
        .cat-input { display: block; width: 100%; margin-bottom: 8px; padding: 9px 12px; border: 1px solid var(--gris-borde); border-radius: 8px; font-size: 0.88rem; font-family: inherit; }
        .cat-input:focus { outline: none; border-color: var(--guinda-claro); }
        .historial-tabla { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .historial-tabla th { text-align: left; padding: 10px 8px; border-bottom: 2px solid var(--guinda); color: var(--guinda); font-size: 0.78rem; text-transform: uppercase; }
        .historial-tabla td { padding: 10px 8px; border-bottom: 1px solid #EDF1F6; }
        .badge-cat { padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; color: white; font-weight: 600; }
        .badge-score { font-weight: 700; }
        .vacio { text-align: center; color: #999; padding: 20px; font-size: 0.9rem; }
        .links { text-align: center; margin-top: 10px; }
        .links a { color: white; text-decoration: none; margin: 0 10px; font-size: 0.9rem; opacity: 0.9; }
        .links a:hover { text-decoration: underline; }
        .footer { text-align: center; color: white; opacity: 0.8; font-size: 0.85rem; padding: 20px; }
        .loading { display: none; text-align: center; color: var(--dorado); padding: 10px; }
        @media (max-width: 720px) { .grid-2 { grid-template-columns: 1fr; } .stats { grid-template-columns: repeat(2, 1fr); } }
    </style>
</head>
<body>
    <div class="cinta-dorada"></div>
    <div class="container">
        <div class="header">
            <img class="logo-unmsm" src="/static/logo-unmsm.png" alt="Universidad Nacional Mayor de San Marcos">
            <h1>Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos</h1>
            <p>Red Neuronal Convolucional (CNN) para la Mesa de Partes Virtual — FISI-UNMSM</p>
            <div class="badge">Decana de América · FISI | Ortiz Herrera</div>
        </div>

        <div class="card">
            <h2>Clasificar Documento (FUT)</h2>
            <textarea id="texto" placeholder="Escriba o pegue aquí el texto de la solicitud o documento administrativo-académico a clasificar..."></textarea>
            <div class="ejemplos-grupo">
                <div class="titulo-grupo">Ejemplos por categoría (haga clic para cargar)</div>
                <div class="ejemplos" id="ejemplosContainer"></div>
            </div>
            <button class="btn-clasificar" onclick="clasificar()">Clasificar Documento</button>
            <div class="loading" id="loading">Procesando con la red neuronal...</div>
            <div class="resultado" id="resultado">
                <h3 id="categoria"></h3>
                <p class="tiempo-info" id="tiempoInfo"></p>
                <p style="font-size:0.9rem; color:#555; margin-bottom:6px;">Nivel de confianza:</p>
                <div class="score-bar"><div class="score-fill" id="scoreFill">0%</div></div>
                <p id="alertaMsg" style="font-size:0.9rem; margin-top:10px;"></p>
                <div id="terminosClave"></div>
            </div>
        </div>

        <div class="card">
            <h2>Métricas del Sistema</h2>
            <div class="stats">
                <div class="stat-box"><div class="num" id="totalDocs">0</div><div class="label">Documentos procesados</div></div>
                <div class="stat-box"><div class="num" id="totalAlertas">0</div><div class="label">Alertas de revisión</div></div>
                <div class="stat-box"><div class="num" id="tiempoProm">0</div><div class="label">Tiempo promedio (ms)</div></div>
                <div class="stat-box"><div class="num" id="numCategorias">5</div><div class="label">Categorías activas</div></div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h2>Distribución por Categoría</h2>
                <div class="chart-wrap"><canvas id="graficoDistribucion"></canvas></div>
            </div>
            <div class="card">
                <h2>Tiempo de Inferencia (últimas clasificaciones)</h2>
                <div class="chart-wrap"><canvas id="graficoTiempos"></canvas></div>
            </div>
        </div>

        <div class="card">
            <h2>Configuración de Categorías</h2>
            <p style="font-size:0.85rem;color:#666;margin-bottom:14px;">El modelo CNN tiene 5 clases fijas por arquitectura; aquí puede renombrarlas sin reentrenar.</p>
            <div id="categoriasForm"></div>
            <button class="btn-secundario" onclick="guardarCategorias()">Guardar cambios</button>
            <p id="categoriasMsg" style="font-size:0.85rem;margin-top:8px;color:#666;"></p>
        </div>

        <div class="card">
            <h2>Historial de Clasificaciones</h2>
            <div id="historialContainer"><p class="vacio">Aún no se han clasificado documentos. Pruebe con un ejemplo.</p></div>
            <a class="btn-exportar" href="/historial/exportar">Exportar historial (CSV)</a>
        </div>

        <div class="links">
            <a href="/docs">Documentación API (Swagger)</a>
            <a href="/historial" target="_blank">Historial (JSON)</a>
            <a href="/metricas" target="_blank">Métricas (JSON)</a>
        </div>

        <div class="footer">
            Universidad Nacional Mayor de San Marcos — Facultad de Ingeniería de Sistemas e Informática<br>
            Tesis de pregrado 2025 | Clasificación de documentos administrativo-académicos mediante CNN
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <script>
        const CATS = ["Certificados de Estudios","Constancias Académicas","Trámites de Convalidación","Trámites de Grados y Títulos","Solicitudes Administrativas Generales"];
        const BADGE_COLORS = ["#731E19","#A54A3F","#B8902E","#1B7A72","#2E5B99"];
        const ejemplos = {
            "Certificados de Estudios": [
                "Solicito la emisión del certificado de estudios correspondiente a los ciclos I al VI de la carrera de Ingeniería de Sistemas, para trámite de homologación en universidad extranjera.",
                "Solicito certificado de estudios oficial con sello de la facultad para presentar en proceso de admisión a maestría.",
                "Requiero copia certificada de mi certificado de estudios, ya que el documento original fue extraviado."
            ],
            "Constancias Académicas": [
                "Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa.",
                "Solicito constancia de notas del ciclo 2025-II para presentar ante mi centro de trabajo.",
                "Pido constancia de orden de mérito correspondiente al décimo ciclo de la carrera de Ingeniería de Sistemas."
            ],
            "Trámites de Convalidación": [
                "Solicito la convalidación del curso de Cálculo I llevado en la Universidad Nacional de Ingeniería, cursado en el ciclo 2023-I.",
                "Solicito convalidación de cursos aprobados en programa de intercambio en la Universidad de Chile.",
                "Pido evaluación y convalidación de la asignatura de Física General II proveniente de traslado externo."
            ],
            "Trámites de Grados y Títulos": [
                "Solicito la inscripción de mi expediente para optar el título profesional de Ingeniero de Sistemas.",
                "Pido programación de fecha de sustentación de tesis para la obtención del título profesional.",
                "Solicito el grado académico de Bachiller en Ingeniería de Sistemas, habiendo cumplido con los requisitos establecidos."
            ],
            "Solicitudes Administrativas Generales": [
                "Solicito la rectificación de mis datos personales en el sistema académico debido a un error en el número de documento de identidad.",
                "Pido la emisión de un duplicado de carné universitario por pérdida del documento original.",
                "Solicito autorización para reserva de matrícula del ciclo 2026-I por motivos de salud."
            ]
        };
        let categoriasEdit = [];
        let chartDistribucion = null;
        let chartTiempos = null;

        function renderEjemplos() {
            const cont = document.getElementById('ejemplosContainer');
            let html = '';
            for (const cat in ejemplos) html += '<button class="ejemplo-btn" onclick="usarEjemplo(\'' + cat + '\')">' + cat + '</button>';
            html += '<button class="ejemplo-btn btn-aleatorio" onclick="ejemploAleatorio()">Ejemplo aleatorio</button>';
            cont.innerHTML = html;
        }
        function usarEjemplo(cat) {
            const arr = ejemplos[cat];
            document.getElementById('texto').value = arr[Math.floor(Math.random() * arr.length)];
        }
        function ejemploAleatorio() {
            const cats = Object.keys(ejemplos);
            usarEjemplo(cats[Math.floor(Math.random() * cats.length)]);
        }
        async function clasificar() {
            const texto = document.getElementById('texto').value.trim();
            if (texto.length < 10) { alert('Escriba un texto de al menos 10 caracteres.'); return; }
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultado').style.display = 'none';
            try {
                const resp = await fetch('/clasificar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texto: texto }) });
                const data = await resp.json();
                document.getElementById('loading').style.display = 'none';
                const resultado = document.getElementById('resultado');
                const score = Math.round(data.score_confianza * 100);
                document.getElementById('categoria').textContent = data.categoria;
                document.getElementById('tiempoInfo').textContent = 'Clasificado en ' + data.tiempo_inferencia_ms + ' ms';
                const fill = document.getElementById('scoreFill');
                fill.style.width = score + '%';
                fill.textContent = score + '%';
                if (data.alerta_revision_manual) {
                    resultado.className = 'resultado alerta'; fill.className = 'score-fill bajo';
                    document.getElementById('alertaMsg').textContent = 'Score inferior al umbral de 0.60. Requiere revisión manual.';
                } else if (score < 75) {
                    resultado.className = 'resultado dudoso'; fill.className = 'score-fill medio';
                    document.getElementById('alertaMsg').textContent = 'Clasificación aceptable pero con confianza moderada.';
                } else {
                    resultado.className = 'resultado exito'; fill.className = 'score-fill alto';
                    document.getElementById('alertaMsg').textContent = 'Clasificación confiable. Documento categorizado automáticamente.';
                }
                const terms = data.terminos_clave || [];
                document.getElementById('terminosClave').innerHTML = terms.length
                    ? '<p style="font-size:0.8rem;color:#666;margin:10px 0 6px;">Términos que más influyeron:</p>' + terms.map(t => '<span class="chip">' + t + '</span>').join('')
                    : '';
                resultado.style.display = 'block';
                actualizarTodo();
            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                alert('Error al clasificar: ' + e);
            }
        }
        function actualizarGraficoDistribucion(dist, cats) {
            const data = cats.map(c => dist[c] || 0);
            if (chartDistribucion) {
                chartDistribucion.data.labels = cats;
                chartDistribucion.data.datasets[0].data = data;
                chartDistribucion.update();
                return;
            }
            chartDistribucion = new Chart(document.getElementById('graficoDistribucion'), {
                type: 'doughnut',
                data: { labels: cats, datasets: [{ data: data, backgroundColor: BADGE_COLORS, borderWidth: 2, borderColor: '#fff' }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10.5 } } } } }
            });
        }
        function actualizarGraficoTiempos(hist) {
            const ultimos = hist.slice(0, 15).slice().reverse();
            const labels = ultimos.map(r => '#' + r.id);
            const data = ultimos.map(r => r.tiempo_inferencia_ms);
            if (chartTiempos) {
                chartTiempos.data.labels = labels;
                chartTiempos.data.datasets[0].data = data;
                chartTiempos.update();
                return;
            }
            chartTiempos = new Chart(document.getElementById('graficoTiempos'), {
                type: 'line',
                data: { labels: labels, datasets: [{ label: 'ms', data: data, borderColor: '#B8902E', backgroundColor: 'rgba(184,144,46,0.15)', tension: 0.3, fill: true }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
            });
        }
        async function cargarCategorias() {
            const resp = await fetch('/categorias', { method: 'PUT' });
            const data = await resp.json();
            categoriasEdit = data.categorias;
            renderCategoriasForm();
        }
        function renderCategoriasForm() {
            const cont = document.getElementById('categoriasForm');
            cont.innerHTML = categoriasEdit.map((c, i) =>
                '<input type="text" class="cat-input" data-idx="' + i + '" value="' + c.replace(/"/g, '&quot;') + '">'
            ).join('');
        }
        async function guardarCategorias() {
            const inputs = document.querySelectorAll('.cat-input');
            const nuevas = Array.from(inputs).map(i => i.value.trim());
            const msg = document.getElementById('categoriasMsg');
            if (nuevas.some(n => !n)) { msg.textContent = 'Ningún nombre puede quedar vacío.'; return; }
            const resp = await fetch('/categorias', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ categorias: nuevas }) });
            const data = await resp.json();
            if (resp.ok) {
                msg.textContent = data.mensaje;
                categoriasEdit = data.categorias;
                CATS.length = 0; CATS.push.apply(CATS, data.categorias);
                actualizarTodo();
            } else {
                msg.textContent = data.detail || 'Error al actualizar categorías.';
            }
        }
        async function actualizarTodo() {
            try {
                const resp = await fetch('/metricas');
                const data = await resp.json();
                document.getElementById('totalDocs').textContent = data.total_documentos_procesados;
                document.getElementById('totalAlertas').textContent = data.documentos_con_alerta;
                document.getElementById('tiempoProm').textContent = data.tiempo_promedio_inferencia_ms;
                document.getElementById('numCategorias').textContent = data.categorias_disponibles.length;
                actualizarGraficoDistribucion(data.distribucion_por_categoria, data.categorias_disponibles);
                const hResp = await fetch('/historial');
                const hist = await hResp.json();
                actualizarHistorial(hist);
                actualizarGraficoTiempos(hist);
            } catch (e) {}
        }
        function actualizarHistorial(hist) {
            const cont = document.getElementById('historialContainer');
            if (hist.length === 0) { cont.innerHTML = '<p class="vacio">Aún no se han clasificado documentos.</p>'; return; }
            let html = '<table class="historial-tabla"><thead><tr><th>#</th><th>Documento</th><th>Categoría</th><th>Score</th><th>Hora</th></tr></thead><tbody>';
            hist.slice(0, 8).forEach(r => {
                const catIdx = CATS.indexOf(r.categoria);
                const color = catIdx >= 0 ? BADGE_COLORS[catIdx] : '#666';
                const score = Math.round(r.score_confianza * 100);
                const scoreColor = r.alerta_revision_manual ? '#E74C3C' : (score < 75 ? '#F39C12' : '#27AE60');
                const hora = r.timestamp.split(' ')[1] || '';
                html += '<tr><td>' + r.id + '</td><td>' + r.texto_fragmento.substring(0, 45) + '...</td><td><span class="badge-cat" style="background:' + color + '">' + r.categoria + '</span></td><td><span class="badge-score" style="color:' + scoreColor + '">' + score + '%</span></td><td>' + hora + '</td></tr>';
            });
            html += '</tbody></table>';
            cont.innerHTML = html;
        }
        renderEjemplos();
        cargarCategorias();
        actualizarTodo();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
