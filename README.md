# Clasificación de razas con Oxford IIIT Pet

Solución integral de la actividad grupal de Redes Neuronales Convolucionales. El proyecto clasifica las 37 razas del Oxford-IIIT Pet Dataset y construye una comparación experimental reproducible entre modelos Fully Connected, CNN propias y transfer learning.

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/enriquemdev/Ai-Oxford-IIIT-Pet-Dataset/blob/main/Actividad_grupal_SCA.ipynb)

## Entrega principal

El archivo `Actividad_grupal_SCA.ipynb` es la entrega autocontenida. Está escrito como notebook e informe técnico ejecutable y cubre:

- análisis exploratorio, distribución de clases y ejemplos;
- corrección documentada del split original `train[:80]` a `train[:80%]`;
- normalización, `tf.data`, semillas y trazabilidad del entorno;
- seis experimentos comparables;
- ablación controlada para medir el efecto de data augmentation;
- callbacks, checkpoints y recuperación sin reentrenar;
- selección por validación, sin consultar test durante el ajuste;
- evaluación final en test, precision, recall, F1 y matriz de confusión;
- análisis de pares de razas confundidas y galería de errores;
- diagnóstico de overfitting o underfitting a partir de las curvas;
- conclusiones automáticas basadas en resultados reales.

## Experimentos

| ID | Modelo | Propósito |
|---|---|---|
| `fc_baseline` | Fully Connected | Baseline obligatorio sin convoluciones |
| `cnn_baseline` | CNN propia sencilla | Medir el aporte de la estructura espacial |
| `cnn_regularized_no_aug` | CNN propia con BN y dropout | Separar regularización de augmentation |
| `cnn_regularized_aug` | Misma CNN con augmentation | Ablación controlada del augmentation |
| `mobilenet_transfer` | MobileNetV2 congelada | Transfer learning con ImageNet |
| `mobilenet_finetuned` | MobileNetV2 parcialmente descongelada | Candidato final para superar 85% |

## Ejecución recomendada en Google Colab

1. Abra `Actividad_grupal_SCA.ipynb` en Colab.
2. Seleccione **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Ejecute todas las celdas en orden.
4. Mantenga `FAST_MODE = False` para la corrida final.
5. Si desea conservar checkpoints entre sesiones, active `USE_GOOGLE_DRIVE = True`.
6. Al finalizar, descargue `entrega_oxford_pet.zip` desde la última celda y exporte el notebook a PDF.

La guía detallada está en [`docs/GUIA_COLAB.md`](docs/GUIA_COLAB.md).

No use las métricas de `FAST_MODE` en la entrega: ese modo solo comprueba que el flujo funciona con una muestra pequeña.

## Ejecución local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab Actividad_grupal_SCA.ipynb
```

En Apple Silicon puede usarse el entorno de TensorFlow compatible con Metal. Para los entrenamientos finales, Colab con GPU ofrece una experiencia más consistente.

## Artefactos generados

La ejecución crea carpetas ignoradas por Git:

```text
models/       mejores checkpoints .keras
artifacts/    historiales, métricas, tablas, figuras e informe de resultados
```

El test se evalúa únicamente después de cerrar la selección con validación. Los resultados no están precargados ni inventados: aparecen al ejecutar el notebook sobre el dataset real.

## Material suministrado

La plantilla original y el enunciado se conservan en `materiales/` para mantener trazabilidad.

## Validación sin TensorFlow

```bash
python3 -m unittest discover -s tests -v
```

La prueba verifica el formato del notebook, la sintaxis de todas las celdas de código, las secciones obligatorias y que no reaparezca el split erróneo de 80 imágenes.

La validación técnica realizada antes de publicar se documenta en [`docs/VALIDACION_TECNICA.md`](docs/VALIDACION_TECNICA.md).
