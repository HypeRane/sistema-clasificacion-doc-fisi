"""Pruebas de la arquitectura CNN y la lógica de inferencia (modelo.py)."""
import torch
import pytest

from modelo import CATEGORIAS, DATOS_ENTRENAMIENTO, CNN_Clasificador, entrenar_modelo, clasificar


@pytest.fixture(scope="module")
def modelo_entrenado():
    return entrenar_modelo()


def test_categorias_son_cinco():
    assert len(CATEGORIAS) == 5
    assert len(set(CATEGORIAS)) == 5  # sin duplicados


def test_datos_entrenamiento_cubren_las_cinco_clases():
    etiquetas = {etiqueta for _, etiqueta in DATOS_ENTRENAMIENTO}
    assert etiquetas == set(range(5))


def test_arquitectura_forward_produce_la_forma_esperada():
    modelo = CNN_Clasificador(vocab_size=100, num_clases=5)
    x = torch.rand(4, 100)
    salida = modelo(x)
    assert salida.shape == (4, 5)


def test_clasificar_devuelve_la_estructura_esperada(modelo_entrenado):
    modelo, vectorizer = modelo_entrenado
    resultado = clasificar(
        "Solicito constancia de matrícula del ciclo 2026-I para trámite de beca externa.",
        modelo, vectorizer,
    )
    assert set(resultado.keys()) == {
        "categoria", "categoria_id", "score_confianza",
        "alerta_revision_manual", "terminos_clave",
    }
    assert resultado["categoria"] in CATEGORIAS
    assert 0 <= resultado["categoria_id"] < len(CATEGORIAS)
    assert 0.0 <= resultado["score_confianza"] <= 1.0
    assert isinstance(resultado["terminos_clave"], list)
    assert resultado["alerta_revision_manual"] == (resultado["score_confianza"] < 0.60)


def test_clasificar_reconoce_la_mayoria_de_sus_propios_ejemplos(modelo_entrenado):
    modelo, vectorizer = modelo_entrenado
    aciertos = sum(
        1 for texto, etiqueta in DATOS_ENTRENAMIENTO
        if clasificar(texto, modelo, vectorizer)["categoria_id"] == etiqueta
    )
    # Con ~8 ejemplos por clase el modelo deberia sobre-ajustar y reconocer
    # la gran mayoria de sus propios textos de entrenamiento.
    assert aciertos / len(DATOS_ENTRENAMIENTO) >= 0.8
