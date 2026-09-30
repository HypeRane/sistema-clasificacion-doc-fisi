"""Configuración compartida de pytest: limpia los artefactos generados en tiempo de ejecución.

La limpieza "antes" ocurre al importarse este módulo, que pytest carga antes de
recolectar/importar los módulos de prueba (y por lo tanto antes de que
`import main` dispare el entrenamiento y la creación de historial.db). Usar un
fixture normal para esto sería demasiado tarde: los fixtures se ejecutan
después de que los tests ya importaron `main`.
"""
import os
import pytest

ARTEFACTOS_GENERADOS = ["historial.db", "categorias.json", "modelo_cnn.pt", "vectorizer.pkl"]


def _borrar_artefactos():
    for nombre in ARTEFACTOS_GENERADOS:
        if os.path.exists(nombre):
            os.remove(nombre)


_borrar_artefactos()


def pytest_sessionfinish(session, exitstatus):
    _borrar_artefactos()
