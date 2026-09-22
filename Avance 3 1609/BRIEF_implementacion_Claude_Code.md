# Brief de implementación — Tesis Fabrizio Ortiz Herrera (FISI-UNMSM)

Pega esto como primer mensaje a Claude Code en VS Code, junto con `main.py` y
`modelo.py` abiertos en el editor. Este brief asume que el documento Word de
la tesis (los 4 capítulos) ya está terminado — el trabajo que queda es el
prototipo.

## Quién soy y qué es esto
Fabrizio Peter Ortiz Herrera, estudiante de décimo ciclo de Ingeniería de
Sistemas, FISI-UNMSM. Tesis de pregrado sobre clasificación automatizada de
documentos administrativo-académicos.

## El pivote de dominio (contexto necesario)
La tesis originalmente clasificaba documentos CLÍNICOS de hospitales. Por
recomendación de la profesora del curso (Ley 29733 — los datos de salud son
sensibles y piden convenio institucional, comité de ética, etc.), el dominio
cambió a documentos ADMINISTRATIVO-ACADÉMICOS de la Mesa de Partes Virtual de
la FISI-UNMSM — el Formato Único de Trámite (FUT). Los 4 capítulos del Word
ya están completamente actualizados a este nuevo dominio. Lo único que falta
es que el CÓDIGO del prototipo (`main.py`, `modelo.py`) siga reflejando el
dominio viejo.

## Las 5 categorías — CAMBIO CRÍTICO Y OBLIGATORIO
El prototipo actual tiene hardcodeadas estas 5 categorías CLÍNICAS que ya NO
aplican:
1. ~~Nota de alta médica~~
2. ~~Registro de admisión~~
3. ~~Informe de laboratorio~~
4. ~~Nota de evolución clínica~~
5. ~~Informe de imagen diagnóstica~~

Deben reemplazarse por estas 5 categorías ADMINISTRATIVO-ACADÉMICAS (ya
están así en todo el Word — Capítulo IV, tablas de requisitos, tabla CRISP-DM,
diagramas, mockup de Swagger):
1. **Certificados de Estudios**
2. **Constancias Académicas** (matrícula, notas, orden de mérito, egresado, ingreso)
3. **Trámites de Convalidación**
4. **Trámites de Grados y Títulos**
5. **Solicitudes Administrativas Generales**

## Decisión técnica ya tomada — NO cambiar
CNN sobre **TEXTO** (TF-IDF + CNN), NO sobre imagen del documento escaneado.
Se evaluó y descartó conscientemente el enfoque de imagen (ResNet/EfficientNet,
estilo benchmark RVL-CDIP) para mantener el pipeline ya construido y no
depender de calidad de escaneo. Esto ya está justificado por escrito en el
Marco Teórico (Capítulo II) citando a Harley et al. (2015) y Xu et al. (2020)
como el enfoque alternativo considerado y descartado.

## Qué hay que hacer en el código

### 1. `modelo.py`
- Reemplazar las 5 categorías (constante `CATEGORIAS` o equivalente)
- Reemplazar los ~35 textos de entrenamiento embebidos (7 por categoría) por
  ~35-50 textos de ejemplo de solicitudes/documentos FUT reales de las 5
  categorías nuevas. Ejemplos de tono a replicar (ya usado en el mockup de
  Swagger del Word): *"Solicito constancia de matrícula del ciclo 2026-I
  para trámite de beca externa."*
- El umbral de confianza (0.60) y la lógica de alerta de revisión manual se
  mantienen igual — solo cambia el dominio del texto, no la arquitectura

### 2. `main.py`
- Actualizar el HTML de la interfaz web: título, ejemplos por categoría (los
  3 ejemplos aleatorios por categoría que se muestran en el frontend),
  cualquier texto de placeholder o encabezado que diga "clínico" o mencione
  hospitales
- Los endpoints (`POST /clasificar`, `GET /historial`, `GET /metricas`,
  `GET /alertas`, `PUT /categorias`) se mantienen igual en su firma y
  comportamiento — solo cambian los textos de ejemplo/documentación que
  contengan

### 3. Nombre del repositorio de GitHub
El repo se llama `sistema-clasificacion-clinica` — el nombre quedó
desactualizado. Sugerido: `sistema-clasificacion-documentos-fisi` o similar.
Actualizar también el README si menciona "clínico" en algún lado.

## Apartado de tecnologías ya escrito (para mejorar, no rehacer desde cero)
El Capítulo IV ya tiene:
- **Tabla 13**: tecnologías utilizadas (10 tecnologías)
- **4.1.2**: comparación de capa lógica (PyTorch vs TF vs Keras), capa de
  datos (PostgreSQL vs MySQL vs SQLite)
- **4.1.3**: comparación de framework API (FastAPI vs Flask vs DRF)
- Diagrama de arquitectura (Figura 7): 3 capas — Presentación (FastAPI),
  Procesamiento (CNN/PyTorch), Datos + Monitoreo

Si quieres mejorarlo, lo natural sería: agregar más profundidad técnica real
una vez que el prototipo esté reconstruido (métricas reales de entrenamiento,
tiempos de inferencia reales, decisiones de arquitectura documentadas con más
detalle), no reescribirlo desde cero.

## Pendiente fuera del código (contexto, no para Claude Code)
- **Figura 6** (diagrama de actividades del CU-01) — nunca se creó, el índice
  la menciona pero el cuerpo del documento no la tiene. Es un diagrama, no
  código.
- **Dataset real**: se está gestionando acceso al archivo digitalizado de la
  Facultad (data aproximada desde 2018) a través de la profesora Fany Sobero
  y el profesor Bustamante (Archivo Central). Mientras no llegue, el dataset
  de entrenamiento sigue siendo sintético/de ejemplo.

## Preferencias de estilo (si Claude Code toca el Word también)
- Times New Roman 12pt, interlineado 1.5, justificado
- Tablas: solo líneas horizontales, sin sombreado
- Notas de tabla/figura: cursiva 11pt, formato "Nota. [texto]. Elaboración propia."
- Si edita el `.docx`, usar python-docx con cuidado de no borrar los
  comentarios de Word de la profesora — o trabajar directo sobre el XML
  (unzip → editar `word/document.xml` → zip) si python-docx los rompe
