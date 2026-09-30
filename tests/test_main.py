"""Pruebas de integración de la API REST (main.py).

Los tests están numerados porque comparten un mismo TestClient / base de datos
SQLite y se acumulan en secuencia (clasifican un documento, luego verifican que
aparezca en el historial, luego renombran categorías, etc.), igual que un flujo
real de uso del sistema.
"""
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

TEXTO_EJEMPLO = "Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa."


def test_01_estado_reporta_sistema_activo():
    r = client.get("/estado")
    assert r.status_code == 200
    assert r.json()["estado"] == "activo"


def test_02_clasificar_rechaza_texto_demasiado_corto():
    r = client.post("/clasificar", json={"texto": "abc"})
    assert r.status_code == 422


def test_03_clasificar_texto_valido_devuelve_resultado_completo():
    r = client.post("/clasificar", json={"texto": TEXTO_EJEMPLO})
    assert r.status_code == 200
    data = r.json()
    for campo in ("categoria", "score_confianza", "alerta_revision_manual",
                  "terminos_clave", "tiempo_inferencia_ms", "timestamp"):
        assert campo in data
    assert data["categoria"] in main.CATEGORIAS
    assert isinstance(data["terminos_clave"], list)


def test_04_historial_incluye_el_documento_clasificado():
    r = client.get("/historial")
    assert r.status_code == 200
    historial = r.json()
    assert len(historial) >= 1
    assert historial[0]["texto_fragmento"].startswith("Solicito constancia de matrícula")


def test_05_historial_filtra_por_categoria_inexistente():
    r = client.get("/historial", params={"categoria": "categoria-que-no-existe"})
    assert r.status_code == 200
    assert r.json() == []


def test_06_metricas_reflejan_el_documento_procesado():
    r = client.get("/metricas")
    assert r.status_code == 200
    data = r.json()
    assert data["total_documentos_procesados"] >= 1
    assert len(data["categorias_disponibles"]) == 5


def test_07_alertas_devuelve_una_lista():
    r = client.get("/alertas")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_08_categorias_sin_body_devuelve_las_activas():
    r = client.put("/categorias")
    assert r.status_code == 200
    assert len(r.json()["categorias"]) == 5


def test_09_categorias_con_longitud_invalida_falla():
    r = client.put("/categorias", json={"categorias": ["Solo", "Dos"]})
    assert r.status_code == 400


def test_10_categorias_renombradas_correctamente():
    nuevas = ["Cat A", "Cat B", "Cat C", "Cat D", "Cat E"]
    r = client.put("/categorias", json={"categorias": nuevas})
    assert r.status_code == 200
    assert r.json()["categorias"] == nuevas

    # el historial ya clasificado debe reetiquetarse con el nuevo nombre
    r_hist = client.get("/historial")
    categorias_en_historial = {registro["categoria"] for registro in r_hist.json()}
    assert categorias_en_historial.issubset(set(nuevas))


def test_11_historial_exportar_devuelve_csv():
    r = client.get("/historial/exportar")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "texto_fragmento" in r.text
