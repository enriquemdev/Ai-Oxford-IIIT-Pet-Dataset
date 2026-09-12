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

## Resultado de la ejecución final

La corrida completa se ejecutó localmente sobre una GPU Apple M5 mediante TensorFlow Metal, usando 2,944 imágenes de entrenamiento, 368 de validación y 368 de test. El modelo se seleccionó antes de abrir test.

| Resultado | Valor |
|---|---:|
| Modelo seleccionado | `mobilenet_finetuned` |
| Validation accuracy | 88.04% |
| Test accuracy | **86.14%** |
| Macro precision | 86.43% |
| Macro recall | 85.81% |
| Macro F1 | 85.10% |
| Brecha train-validation en la mejor época | 2.99 puntos |

El objetivo de 85% se cumplió sin evidencia fuerte de overfitting o underfitting según el criterio documentado. El informe académico final de 10 páginas está en [`output/pdf/informe_tecnico_oxford_pet.pdf`](output/pdf/informe_tecnico_oxford_pet.pdf).

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
/path/to/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab Actividad_grupal_SCA.ipynb
```

En Apple Silicon, `requirements.txt` instala una combinación compatible de TensorFlow y `tensorflow-metal`. Verifique la aceleración antes de entrenar:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

La salida debe incluir `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`. Para los entrenamientos finales, Colab con GPU sigue siendo una alternativa si se necesita una sesión más consistente o prolongada.

## Artefactos generados

La ejecución crea carpetas ignoradas por Git:

```text
models/       mejores checkpoints .keras
artifacts/    historiales, métricas, tablas, figuras e informe de resultados
```

El test se evalúa únicamente después de cerrar la selección con validación. El notebook principal conserva las salidas de la corrida completa; los resultados provienen del dataset real y pueden regenerarse a partir de los checkpoints e historiales locales.

## Material suministrado

La plantilla original y el enunciado se conservan en `materiales/` para mantener trazabilidad.

## Validación sin TensorFlow

```bash
python3 -m unittest discover -s tests -v
```

La prueba verifica el formato del notebook, la sintaxis de todas las celdas de código, las secciones obligatorias y que no reaparezca el split erróneo de 80 imágenes.

La validación técnica realizada antes de publicar se documenta en [`docs/VALIDACION_TECNICA.md`](docs/VALIDACION_TECNICA.md).
