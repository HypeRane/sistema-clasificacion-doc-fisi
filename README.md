# Sistema de Clasificación Automatizada de Documentos Administrativo-Académicos

Sistema basado en **Redes Neuronales Convolucionales (CNN)** para la clasificación automatizada de documentos administrativo-académicos en español, orientado a la Mesa de Partes Virtual (Formato Único de Trámite — FUT) de la FISI-UNMSM.

**Universidad Nacional Mayor de San Marcos — Facultad de Ingeniería de Sistemas e Informática**
Tesis de pregrado | Autor: Fabrizio Peter Ortiz Herrera

---

## Descripción

Este sistema utiliza una arquitectura CNN entrenada sobre texto de solicitudes FUT en español para clasificar automáticamente documentos administrativo-académicos en cinco categorías:

- Certificados de Estudios
- Constancias Académicas
- Trámites de Convalidación
- Trámites de Grados y Títulos
- Solicitudes Administrativas Generales

El sistema expone sus funcionalidades mediante una **API REST desarrollada con FastAPI**, con documentación interactiva automática (Swagger UI) y una interfaz web para la clasificación de documentos. Además de la categoría y el score de confianza, cada clasificación devuelve los **términos TF-IDF que más influyeron** en la predicción, como elemento de explicabilidad del modelo.

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.13 |
| Modelo CNN | PyTorch |
| API REST | FastAPI + Uvicorn |
| Vectorización | scikit-learn (TF-IDF) |
| Persistencia | SQLite |
| Visualización | Chart.js |
| Documentación | Swagger UI (OpenAPI) |
| Contenerización | Docker + Docker Compose |
| Pruebas | pytest |

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/HypeRane/sistema-clasificacion-documentos-fisi.git
cd sistema-clasificacion-documentos-fisi
```

### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el sistema

```bash
python main.py
```

El sistema entrenará automáticamente el modelo CNN en el primer arranque y quedará disponible en:

- **Interfaz web:** http://localhost:8000
- **Documentación API (Swagger):** http://localhost:8000/docs

---

## Ejecución con Docker

```bash
docker compose up --build
```

El servicio queda disponible en `http://localhost:8000`. Por defecto no se declara un volumen: el modelo se reentrena en segundos en cada arranque del contenedor (mismo comportamiento que en local cuando no existen `modelo_cnn.pt`/`vectorizer.pkl`). Para conservar el historial y el modelo entre recreaciones del contenedor, descomentar el volumen indicado en `docker-compose.yml`.

---

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Incluye pruebas de la arquitectura CNN y la lógica de inferencia (`tests/test_modelo.py`) y pruebas de integración de todos los endpoints de la API (`tests/test_main.py`).

---

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/clasificar` | Clasifica un documento a partir de texto pegado |
| `POST` | `/clasificar-archivo` | Clasifica un documento a partir de un archivo subido (.pdf o .txt) |
| `GET` | `/historial` | Historial de documentos procesados |
| `GET` | `/historial/exportar` | Exporta el historial completo como CSV |
| `GET` | `/metricas` | Métricas de rendimiento del sistema |
| `GET` | `/alertas` | Documentos que requieren revisión manual |
| `PUT` | `/categorias` | Sin cuerpo: consulta las categorías activas. Con cuerpo `{"categorias": [...]}`: las renombra (el número de clases está fijado por el modelo entrenado) |

---

## Arquitectura

El sistema sigue una arquitectura en tres capas:

- **Capa de presentación:** API REST con FastAPI + interfaz web con gráficos (Chart.js)
- **Capa de procesamiento:** Motor CNN con PyTorch
- **Capa de datos:** Historial de predicciones persistido en SQLite (`historial.db`) y configuración de categorías en `categorias.json`

---

## Umbral de confianza

El sistema aplica la función Softmax en la capa final de la CNN para generar un **score de confianza** (0 a 1). Los documentos con score inferior a **0.60** se marcan automáticamente para revisión manual.

---

## Licencia

Proyecto académico desarrollado con fines de investigación para la tesis de pregrado en la UNMSM.
