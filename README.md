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

El sistema expone sus funcionalidades mediante una **API REST desarrollada con FastAPI**, con documentación interactiva automática (Swagger UI) y una interfaz web para la clasificación de documentos.

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| Modelo CNN | PyTorch |
| API REST | FastAPI + Uvicorn |
| Vectorización | scikit-learn (TF-IDF) |
| Documentación | Swagger UI (OpenAPI) |

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

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/clasificar` | Clasifica un documento administrativo-académico |
| `GET` | `/historial` | Historial de documentos procesados |
| `GET` | `/metricas` | Métricas de rendimiento del sistema |
| `GET` | `/alertas` | Documentos que requieren revisión manual |
| `PUT` | `/categorias` | Categorías de clasificación configuradas |

---

## Arquitectura

El sistema sigue una arquitectura en tres capas:

- **Capa de presentación:** API REST con FastAPI + interfaz web
- **Capa de procesamiento:** Motor CNN con PyTorch
- **Capa de datos:** Historial de predicciones y métricas

---

## Umbral de confianza

El sistema aplica la función Softmax en la capa final de la CNN para generar un **score de confianza** (0 a 1). Los documentos con score inferior a **0.60** se marcan automáticamente para revisión manual.

---

## Licencia

Proyecto académico desarrollado con fines de investigación para la tesis de pregrado en la UNMSM.
