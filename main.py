"""
main.py
API REST del Sistema de Clasificación Automatizada de Documentos Clínicos.
Universidad Nacional Mayor de San Marcos — FISI
Autor: Ortiz Herrera, Fabrizio Peter
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from modelo import cargar_modelo, clasificar, CATEGORIAS
import uvicorn

# ── Inicializar aplicación ────────────────────────────────────────────────────
app = FastAPI(
    title="Sistema de Clasificación Automatizada de Documentos Clínicos",
    description="""
## Sistema CNN para Archivos Hospitalarios de Lima

Este sistema utiliza una **Red Neuronal Convolucional (CNN)** entrenada sobre
documentos clínicos en español para clasificar automáticamente expedientes
hospitalarios en cinco categorías:

- Nota de alta médica
- Registro de admisión
- Informe de laboratorio
- Nota de evolución clínica
- Informe de imagen diagnóstica

**Universidad Nacional Mayor de San Marcos — FISI | Tesis de pregrado 2025**
    """,
    version="1.0.0",
    contact={
        "name": "Ortiz Herrera, Fabrizio Peter",
        "email": "fabrizio.ortiz@unmsm.edu.pe"
    }
)

# ── Cargar modelo al iniciar ──────────────────────────────────────────────────
print("Iniciando sistema...")
modelo_cnn, vectorizer = cargar_modelo()
print("Sistema listo.")

# ── Historial en memoria (prototipo) ──────────────────────────────────────────
historial = []

# ── Esquemas de datos ─────────────────────────────────────────────────────────
class DocumentoEntrada(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        description="Texto del documento clínico a clasificar",
        json_schema_extra={"example": "Paciente dado de alta en buen estado general. Diagnóstico: hipertensión arterial controlada. Medicación: enalapril 10mg."}
    )

class ResultadoClasificacion(BaseModel):
    categoria: str
    score_confianza: float
    alerta_revision_manual: bool
    timestamp: str

class RegistroHistorial(BaseModel):
    id: int
    texto_fragmento: str
    categoria: str
    score_confianza: float
    alerta_revision_manual: bool
    timestamp: str

class MetricasSistema(BaseModel):
    total_documentos_procesados: int
    documentos_con_alerta: int
    porcentaje_alertas: float
    categorias_disponibles: list

class AlertaDetalle(BaseModel):
    id: int
    texto_fragmento: str
    score_confianza: float
    categoria_asignada: str
    timestamp: str


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ WEB — Página principal
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse, tags=["Interfaz"], include_in_schema=False)
def interfaz_web():
    """Interfaz web visual del sistema de clasificación."""
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Clasificación de Documentos Clínicos — UNMSM</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1F3864 0%, #2E5B99 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .header {
            text-align: center;
            color: white;
            padding: 30px 20px 20px;
        }
        .header h1 { font-size: 1.8rem; margin-bottom: 8px; font-weight: 700; }
        .header p { font-size: 0.95rem; opacity: 0.9; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            margin-top: 10px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .card h2 {
            font-size: 1.2rem;
            color: #1F3864;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 14px;
            border: 2px solid #E0E6ED;
            border-radius: 10px;
            font-size: 0.95rem;
            font-family: inherit;
            resize: vertical;
            transition: border 0.2s;
        }
        textarea:focus { outline: none; border-color: #2E5B99; }
        .ejemplos {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 12px 0;
        }
        .ejemplo-btn {
            background: #EBF0F8;
            border: 1px solid #C9D6E8;
            color: #1F3864;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .ejemplo-btn:hover { background: #2E5B99; color: white; }
        .btn-clasificar {
            width: 100%;
            background: linear-gradient(135deg, #1F3864, #2E5B99);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 14px;
            transition: transform 0.1s;
        }
        .btn-clasificar:hover { transform: translateY(-2px); }
        .btn-clasificar:active { transform: translateY(0); }
        .resultado {
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            display: none;
            animation: fadeIn 0.4s;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; } }
        .resultado.exito { background: #EAF7EC; border: 2px solid #27AE60; }
        .resultado.alerta { background: #FEF9E7; border: 2px solid #F39C12; }
        .resultado h3 { font-size: 1.3rem; margin-bottom: 12px; }
        .resultado.exito h3 { color: #1D6A35; }
        .resultado.alerta h3 { color: #935A0A; }
        .score-bar {
            background: #E0E6ED;
            border-radius: 20px;
            height: 24px;
            margin: 10px 0;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #27AE60, #2ECC71);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
            transition: width 0.8s ease;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }
        .stat-box {
            background: #F8FAFC;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            border: 1px solid #E0E6ED;
        }
        .stat-box .num { font-size: 1.8rem; font-weight: 700; color: #1F3864; }
        .stat-box .label { font-size: 0.8rem; color: #666; margin-top: 4px; }
        .links {
            text-align: center;
            margin-top: 10px;
        }
        .links a {
            color: white;
            text-decoration: none;
            margin: 0 10px;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .links a:hover { text-decoration: underline; }
        .footer {
            text-align: center;
            color: white;
            opacity: 0.8;
            font-size: 0.85rem;
            padding: 20px;
        }
        .loading { display: none; text-align: center; color: #2E5B99; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sistema de Clasificación Automatizada de Documentos Clínicos</h1>
            <p>Red Neuronal Convolucional (CNN) para Archivos Hospitalarios de Lima</p>
            <div class="badge">UNMSM — FISI | Ortiz Herrera</div>
        </div>

        <div class="card">
            <h2>Clasificar Documento Clínico</h2>
            <textarea id="texto" placeholder="Escriba o pegue aquí el texto del documento clínico a clasificar..."></textarea>

            <div class="ejemplos">
                <button class="ejemplo-btn" onclick="usarEjemplo(0)">Ejemplo: Alta médica</button>
                <button class="ejemplo-btn" onclick="usarEjemplo(1)">Ejemplo: Laboratorio</button>
                <button class="ejemplo-btn" onclick="usarEjemplo(2)">Ejemplo: Admisión</button>
                <button class="ejemplo-btn" onclick="usarEjemplo(3)">Ejemplo: Imagen</button>
                <button class="ejemplo-btn" onclick="usarEjemplo(4)">Ejemplo: Evolución</button>
            </div>

            <button class="btn-clasificar" onclick="clasificar()">Clasificar Documento</button>
            <div class="loading" id="loading">Procesando con la red neuronal...</div>

            <div class="resultado" id="resultado">
                <h3 id="categoria"></h3>
                <p style="font-size:0.9rem; color:#555; margin-bottom:8px;">Nivel de confianza de la clasificación:</p>
                <div class="score-bar">
                    <div class="score-fill" id="scoreFill">0%</div>
                </div>
                <p id="alertaMsg" style="font-size:0.9rem; margin-top:10px;"></p>
            </div>
        </div>

        <div class="card">
            <h2>Métricas del Sistema</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="num" id="totalDocs">0</div>
                    <div class="label">Documentos procesados</div>
                </div>
                <div class="stat-box">
                    <div class="num" id="totalAlertas">0</div>
                    <div class="label">Alertas de revisión</div>
                </div>
                <div class="stat-box">
                    <div class="num" id="numCategorias">5</div>
                    <div class="label">Categorías activas</div>
                </div>
            </div>
        </div>

        <div class="links">
            <a href="/docs">Documentación API (Swagger)</a>
            <a href="/historial" target="_blank">Ver historial (JSON)</a>
            <a href="/metricas" target="_blank">Ver métricas (JSON)</a>
        </div>

        <div class="footer">
            Universidad Nacional Mayor de San Marcos — Facultad de Ingeniería de Sistemas e Informática<br>
            Tesis de pregrado 2025 | Clasificación de documentos clínicos mediante CNN
        </div>
    </div>

    <script>
        const ejemplos = [
            "Paciente dado de alta en buen estado general. Diagnóstico: hipertensión arterial controlada. Medicación: enalapril 10mg. Se indica control ambulatorio en 7 días.",
            "Hemograma completo: Hb 11.2 g/dL, leucocitos 8500/mm3, plaquetas 220000/mm3. Glucosa en ayunas 126 mg/dL. Resultado dentro de rangos normales.",
            "Paciente ingresa por emergencia con cuadro de dolor abdominal agudo de 6 horas de evolución. Sin antecedentes quirúrgicos relevantes.",
            "Radiografía de tórax: cardiomegalia leve. Sin derrame pleural. Infiltrado basal derecho compatible con proceso neumónico en resolución.",
            "Evolución día 2: paciente refiere mejoría del dolor. Afebril. Herida quirúrgica limpia sin signos de infección. Tolera dieta blanda."
        ];

        function usarEjemplo(i) {
            document.getElementById('texto').value = ejemplos[i];
        }

        async function clasificar() {
            const texto = document.getElementById('texto').value.trim();
            if (texto.length < 10) {
                alert('Por favor escriba un texto de al menos 10 caracteres.');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultado').style.display = 'none';

            try {
                const resp = await fetch('/clasificar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ texto: texto })
                });
                const data = await resp.json();

                document.getElementById('loading').style.display = 'none';

                const resultado = document.getElementById('resultado');
                const score = Math.round(data.score_confianza * 100);

                document.getElementById('categoria').textContent = data.categoria;
                document.getElementById('scoreFill').style.width = score + '%';
                document.getElementById('scoreFill').textContent = score + '%';

                if (data.alerta_revision_manual) {
                    resultado.className = 'resultado alerta';
                    document.getElementById('alertaMsg').textContent =
                        '⚠ El score es inferior al umbral de 0.60. Este documento requiere revisión manual.';
                } else {
                    resultado.className = 'resultado exito';
                    document.getElementById('alertaMsg').textContent =
                        '✓ Clasificación confiable. El documento fue categorizado automáticamente.';
                }
                resultado.style.display = 'block';

                actualizarMetricas();
            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                alert('Error al clasificar: ' + e);
            }
        }

        async function actualizarMetricas() {
            try {
                const resp = await fetch('/metricas');
                const data = await resp.json();
                document.getElementById('totalDocs').textContent = data.total_documentos_procesados;
                document.getElementById('totalAlertas').textContent = data.documentos_con_alerta;
                document.getElementById('numCategorias').textContent = data.categorias_disponibles.length;
            } catch (e) {}
        }

        actualizarMetricas();
    </script>
</body>
</html>
    """


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS DE LA API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/estado", tags=["Sistema"])
def estado():
    """Verifica que el sistema está activo."""
    return {
        "sistema": "Clasificación Automatizada de Documentos Clínicos",
        "estado": "activo",
        "version": "1.0.0",
        "documentacion": "/docs"
    }


@app.post("/clasificar", response_model=ResultadoClasificacion, tags=["Clasificación"])
def clasificar_documento(documento: DocumentoEntrada):
    """
    Clasifica un documento clínico en una de las cinco categorías definidas.

    - Recibe el **texto** del documento clínico
    - Devuelve la **categoría asignada**, el **score de confianza** (0-1)
    - Si el score es menor a 0.60, genera una **alerta de revisión manual**
    """
    try:
        resultado = clasificar(documento.texto, modelo_cnn, vectorizer)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        registro = {
            "id": len(historial) + 1,
            "texto_fragmento": documento.texto[:80] + "..." if len(documento.texto) > 80 else documento.texto,
            "categoria": resultado["categoria"],
            "score_confianza": resultado["score_confianza"],
            "alerta_revision_manual": resultado["alerta_revision_manual"],
            "timestamp": timestamp
        }
        historial.append(registro)

        return ResultadoClasificacion(
            categoria=resultado["categoria"],
            score_confianza=resultado["score_confianza"],
            alerta_revision_manual=resultado["alerta_revision_manual"],
            timestamp=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")


@app.get("/historial", response_model=list[RegistroHistorial], tags=["Historial"])
def obtener_historial(
    categoria: Optional[str] = None,
    solo_alertas: Optional[bool] = False
):
    """
    Devuelve el historial de documentos procesados.

    - Filtra por **categoría** (opcional)
    - Filtra por **documentos con alerta** (opcional)
    """
    resultado = historial.copy()

    if categoria:
        resultado = [r for r in resultado if categoria.lower() in r["categoria"].lower()]

    if solo_alertas:
        resultado = [r for r in resultado if r["alerta_revision_manual"]]

    return resultado


@app.get("/metricas", response_model=MetricasSistema, tags=["Monitoreo"])
def obtener_metricas():
    """
    Devuelve métricas de rendimiento del sistema:

    - Total de documentos procesados
    - Documentos con alerta de revisión manual
    - Porcentaje de alertas
    - Categorías disponibles
    """
    total = len(historial)
    alertas = sum(1 for r in historial if r["alerta_revision_manual"])
    porcentaje = round((alertas / total * 100), 2) if total > 0 else 0.0

    return MetricasSistema(
        total_documentos_procesados=total,
        documentos_con_alerta=alertas,
        porcentaje_alertas=porcentaje,
        categorias_disponibles=CATEGORIAS
    )


@app.get("/alertas", response_model=list[AlertaDetalle], tags=["Monitoreo"])
def obtener_alertas():
    """
    Devuelve la lista de documentos que requieren **revisión manual**
    (score de confianza inferior al umbral de 0.60).
    """
    alertas = [
        AlertaDetalle(
            id=r["id"],
            texto_fragmento=r["texto_fragmento"],
            score_confianza=r["score_confianza"],
            categoria_asignada=r["categoria"],
            timestamp=r["timestamp"]
        )
        for r in historial if r["alerta_revision_manual"]
    ]
    return alertas


@app.put("/categorias", tags=["Administración"])
def obtener_categorias():
    """
    Devuelve las categorías de clasificación actualmente configuradas.
    """
    return {
        "categorias": CATEGORIAS,
        "total": len(CATEGORIAS),
        "mensaje": "Categorías activas en el sistema"
    }


# ── Arrancar servidor ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
