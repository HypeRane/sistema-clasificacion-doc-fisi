"""
main.py
API REST del Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos.
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
import time

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
    version="1.1.0",
    contact={"name": "Ortiz Herrera, Fabrizio Peter", "email": "fabrizio.ortiz@unmsm.edu.pe"}
)

print("Iniciando sistema...")
modelo_cnn, vectorizer = cargar_modelo()
print("Sistema listo.")

historial = []
tiempos_inferencia = []


class DocumentoEntrada(BaseModel):
    texto: str = Field(..., min_length=10,
        description="Texto de la solicitud o documento administrativo-académico a clasificar",
        json_schema_extra={"example": "Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa."})

class ResultadoClasificacion(BaseModel):
    categoria: str
    score_confianza: float
    alerta_revision_manual: bool
    tiempo_inferencia_ms: float
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
    distribucion_por_categoria: dict
    tiempo_promedio_inferencia_ms: float
    categorias_disponibles: list

class AlertaDetalle(BaseModel):
    id: int
    texto_fragmento: str
    score_confianza: float
    categoria_asignada: str
    timestamp: str


@app.get("/", response_class=HTMLResponse, tags=["Interfaz"], include_in_schema=False)
def interfaz_web():
    """Interfaz web visual del sistema de clasificación."""
    return HTML_INTERFAZ


@app.get("/estado", tags=["Sistema"])
def estado():
    """Verifica que el sistema está activo."""
    return {"sistema": "Clasificación Automatizada de Documentos Administrativo-Académicos",
            "estado": "activo", "version": "1.1.0", "documentacion": "/docs"}


@app.post("/clasificar", response_model=ResultadoClasificacion, tags=["Clasificación"])
def clasificar_documento(documento: DocumentoEntrada):
    """
    Clasifica un documento administrativo-académico en una de las cinco categorías definidas.

    - Recibe el **texto** de la solicitud o documento
    - Devuelve la **categoría asignada**, el **score de confianza** (0-1)
    - Incluye el **tiempo de inferencia** en milisegundos
    - Si el score es menor a 0.60, genera una **alerta de revisión manual**
    """
    try:
        inicio = time.perf_counter()
        resultado = clasificar(documento.texto, modelo_cnn, vectorizer)
        tiempo_ms = round((time.perf_counter() - inicio) * 1000, 2)
        tiempos_inferencia.append(tiempo_ms)
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
            tiempo_inferencia_ms=tiempo_ms,
            timestamp=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al clasificar: {str(e)}")


@app.get("/historial", response_model=list[RegistroHistorial], tags=["Historial"])
def obtener_historial(categoria: Optional[str] = None, solo_alertas: Optional[bool] = False):
    """Devuelve el historial de documentos procesados."""
    resultado = historial.copy()
    if categoria:
        resultado = [r for r in resultado if categoria.lower() in r["categoria"].lower()]
    if solo_alertas:
        resultado = [r for r in resultado if r["alerta_revision_manual"]]
    return list(reversed(resultado))


@app.get("/metricas", response_model=MetricasSistema, tags=["Monitoreo"])
def obtener_metricas():
    """Devuelve métricas de rendimiento del sistema."""
    total = len(historial)
    alertas = sum(1 for r in historial if r["alerta_revision_manual"])
    porcentaje = round((alertas / total * 100), 2) if total > 0 else 0.0
    distribucion = {cat: 0 for cat in CATEGORIAS}
    for r in historial:
        distribucion[r["categoria"]] += 1
    tiempo_prom = round(sum(tiempos_inferencia) / len(tiempos_inferencia), 2) if tiempos_inferencia else 0.0
    return MetricasSistema(
        total_documentos_procesados=total,
        documentos_con_alerta=alertas,
        porcentaje_alertas=porcentaje,
        distribucion_por_categoria=distribucion,
        tiempo_promedio_inferencia_ms=tiempo_prom,
        categorias_disponibles=CATEGORIAS
    )


@app.get("/alertas", response_model=list[AlertaDetalle], tags=["Monitoreo"])
def obtener_alertas():
    """Devuelve la lista de documentos que requieren revisión manual."""
    return [AlertaDetalle(id=r["id"], texto_fragmento=r["texto_fragmento"],
            score_confianza=r["score_confianza"], categoria_asignada=r["categoria"],
            timestamp=r["timestamp"]) for r in historial if r["alerta_revision_manual"]]


@app.put("/categorias", tags=["Administración"])
def obtener_categorias():
    """Devuelve las categorías de clasificación actualmente configuradas."""
    return {"categorias": CATEGORIAS, "total": len(CATEGORIAS), "mensaje": "Categorías activas en el sistema"}


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
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #1F3864 0%, #2E5B99 100%); min-height: 100vh; padding: 20px; color: #333; }
        .container { max-width: 1050px; margin: 0 auto; }
        .header { text-align: center; color: white; padding: 24px 20px 16px; }
        .header h1 { font-size: 1.7rem; margin-bottom: 8px; font-weight: 700; }
        .header p { font-size: 0.95rem; opacity: 0.9; }
        .badge { display: inline-block; background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; margin-top: 10px; }
        .card { background: white; border-radius: 16px; padding: 26px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px; }
        .card h2 { font-size: 1.15rem; color: #1F3864; margin-bottom: 16px; }
        textarea { width: 100%; min-height: 110px; padding: 14px; border: 2px solid #E0E6ED; border-radius: 10px; font-size: 0.95rem; font-family: inherit; resize: vertical; transition: border 0.2s; }
        textarea:focus { outline: none; border-color: #2E5B99; }
        .ejemplos-grupo { margin: 14px 0; }
        .titulo-grupo { font-size: 0.78rem; color: #888; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .ejemplos { display: flex; flex-wrap: wrap; gap: 8px; }
        .ejemplo-btn { background: #EBF0F8; border: 1px solid #C9D6E8; color: #1F3864; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
        .ejemplo-btn:hover { background: #2E5B99; color: white; }
        .btn-aleatorio { background: #F39C12; border-color: #E67E22; color: white; font-weight: 600; }
        .btn-aleatorio:hover { background: #E67E22; }
        .btn-clasificar { width: 100%; background: linear-gradient(135deg, #1F3864, #2E5B99); color: white; border: none; padding: 14px; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 14px; transition: transform 0.1s; }
        .btn-clasificar:hover { transform: translateY(-2px); }
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
        .score-bar { background: #E0E6ED; border-radius: 20px; height: 26px; margin: 10px 0; overflow: hidden; }
        .score-fill { height: 100%; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-size: 0.82rem; font-weight: 600; transition: width 0.8s ease; width: 0%; }
        .score-fill.alto { background: linear-gradient(90deg, #27AE60, #2ECC71); }
        .score-fill.medio { background: linear-gradient(90deg, #F39C12, #F1C40F); }
        .score-fill.bajo { background: linear-gradient(90deg, #E74C3C, #EC7063); }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
        .stat-box { background: #F8FAFC; border-radius: 10px; padding: 16px; text-align: center; border: 1px solid #E0E6ED; }
        .stat-box .num { font-size: 1.7rem; font-weight: 700; color: #1F3864; }
        .stat-box .label { font-size: 0.78rem; color: #666; margin-top: 4px; }
        .dist-row { margin-bottom: 12px; }
        .dist-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; color: #444; }
        .dist-label .cat-nombre { font-weight: 600; }
        .dist-bar-bg { background: #EDF1F6; border-radius: 8px; height: 20px; overflow: hidden; }
        .dist-bar-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-size: 0.72rem; font-weight: 600; min-width: 30px; }
        .c0 { background: #1F3864; } .c1 { background: #2E5B99; } .c2 { background: #27AE60; } .c3 { background: #8E44AD; } .c4 { background: #E67E22; }
        .historial-tabla { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .historial-tabla th { text-align: left; padding: 10px 8px; border-bottom: 2px solid #1F3864; color: #1F3864; font-size: 0.78rem; text-transform: uppercase; }
        .historial-tabla td { padding: 10px 8px; border-bottom: 1px solid #EDF1F6; }
        .badge-cat { padding: 3px 8px; border-radius: 6px; font-size: 0.72rem; color: white; font-weight: 600; }
        .badge-score { font-weight: 700; }
        .vacio { text-align: center; color: #999; padding: 20px; font-size: 0.9rem; }
        .links { text-align: center; margin-top: 10px; }
        .links a { color: white; text-decoration: none; margin: 0 10px; font-size: 0.9rem; opacity: 0.9; }
        .links a:hover { text-decoration: underline; }
        .footer { text-align: center; color: white; opacity: 0.8; font-size: 0.85rem; padding: 20px; }
        .loading { display: none; text-align: center; color: #2E5B99; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos</h1>
            <p>Red Neuronal Convolucional (CNN) para la Mesa de Partes Virtual — FISI-UNMSM</p>
            <div class="badge">UNMSM — FISI | Ortiz Herrera</div>
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

        <div class="card">
            <h2>Distribución por Categoría</h2>
            <div id="distribucion"></div>
        </div>

        <div class="card">
            <h2>Historial de Clasificaciones</h2>
            <div id="historialContainer"><p class="vacio">Aún no se han clasificado documentos. Pruebe con un ejemplo.</p></div>
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

    <script>
        const CATS = ["Certificados de Estudios","Constancias Académicas","Trámites de Convalidación","Trámites de Grados y Títulos","Solicitudes Administrativas Generales"];
        const COLORS = ["c0","c1","c2","c3","c4"];
        const BADGE_COLORS = ["#1F3864","#2E5B99","#27AE60","#8E44AD","#E67E22"];
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
                resultado.style.display = 'block';
                actualizarTodo();
            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                alert('Error al clasificar: ' + e);
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
                const total = data.total_documentos_procesados;
                const dist = data.distribucion_por_categoria;
                let distHtml = '';
                CATS.forEach((cat, i) => {
                    const count = dist[cat] || 0;
                    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                    distHtml += '<div class="dist-row"><div class="dist-label"><span class="cat-nombre">' + cat + '</span><span>' + count + ' doc(s) — ' + pct + '%</span></div><div class="dist-bar-bg"><div class="dist-bar-fill ' + COLORS[i] + '" style="width:' + pct + '%">' + (pct > 8 ? pct + '%' : '') + '</div></div></div>';
                });
                document.getElementById('distribucion').innerHTML = distHtml;
                const hResp = await fetch('/historial');
                actualizarHistorial(await hResp.json());
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
        actualizarTodo();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
