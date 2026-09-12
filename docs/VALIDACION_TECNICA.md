# Validación técnica de la solución

## Corrida académica final

El 11 de septiembre de 2026 se ejecutó el notebook completo en una Mac Apple M5 con 16 GB de memoria, Python 3.12.14, TensorFlow 2.18.1, TensorFlow Datasets 4.9.10 y aceleración `tensorflow-metal` 1.2.0. La ejecución terminó sin errores en las 25 celdas de código y produjo seis checkpoints y 27 artefactos.

El modelo `mobilenet_finetuned` fue seleccionado exclusivamente con validation (88.04%). En la evaluación final obtuvo 86.14% de accuracy, 86.43% de precision macro, 85.81% de recall macro y 85.10% de macro F1. La brecha train-validation en el mejor checkpoint fue 2.99 puntos, por lo que el objetivo de 85% se considera cumplido sin evidencia fuerte de overfitting o underfitting según el criterio definido.

La ejecución también generó el informe final `output/pdf/informe_tecnico_oxford_pet.pdf`, con 10 páginas, y el paquete local `entrega_oxford_pet.zip`.

## Alcance validado

La solución se verificó en una máquina Linux x86 64 con 9 CPU y 16 GB de RAM, sin GPU. Esta validación comprueba funcionamiento e integración; no pretende sustituir el entrenamiento académico final en Colab.

## Controles realizados

| Control | Resultado |
|---|---|
| Formato JSON y nbformat 4 | Correcto |
| Sintaxis de todas las celdas Python | Correcta |
| Descarga y lectura real de Oxford-IIIT Pet | Correcta |
| Split porcentual y cardinalidades | Correctos |
| Resize a 160 por 160 y normalización a [0, 1] | Correctos |
| Construcción de los seis modelos | Correcta |
| Descarga de pesos ImageNet de MobileNetV2 | Correcta |
| Forward pass de todos los modelos | Correcto |
| Guardado y recarga de modelo `.keras` | Correctos |
| Entrenamiento corto de seis experimentos | Correcto |
| Fine tuning desde el checkpoint congelado | Correcto |
| Selección previa a test | Correcta |
| Classification report con clases ausentes en modo rápido | Correcto |
| Matriz de confusión y análisis de errores | Correctos |
| Generación de 27 artefactos y ZIP final | Correcta |
| Pruebas automatizadas del repositorio | 6 de 6 aprobadas |

## Modo de prueba

La validación integral utilizó `FAST_MODE = True`, 2 épocas por experimento y subconjuntos reducidos. Este modo permitió recorrer el notebook de inicio a fin y detectar problemas de integración sin consumir una sesión larga.

Las métricas de esa corrida no se incluyen porque no representan el experimento solicitado. La corrida académica posterior utilizó `FAST_MODE = False` y la GPU Metal local.

## Dependencias verificadas

La validación inicial se probó con TensorFlow 2.20, TensorFlow Datasets 4.9.9 y Keras 3. La corrida final en Apple Silicon utilizó TensorFlow 2.18.1 y `tensorflow-metal` 1.2.0. En esa plataforma se fijan Protobuf 5.29, `tensorflow-metadata` 1.17.3 y `googleapis-common-protos` 1.66.0 para mantener compatibilidad; las demás plataformas conservan TensorFlow 2.16-2.20 y Protobuf 6.

## Matriz de cobertura académica

| Requisito | Evidencia en el notebook |
|---|---|
| EDA y distribución de clases | Secciones 2 y 3 |
| Cuatro o más modelos CNN | Secciones 7 a 11 |
| Arquitectura propuesta por estudiantes | CNN regularizada de las secciones 8 y 9 |
| Fully Connected como quinto modelo | Sección 6 |
| Data augmentation e impacto | Ablación de secciones 8, 9 y 16 |
| Optimización y regularización | Diseño experimental, BN, Dropout, LR y callbacks |
| Guardar el mejor modelo | `ModelCheckpoint` por `val_accuracy` |
| Evitar reentrenamiento | Carga automática de checkpoints existentes |
| Precision y recall por clase | Sección 14 |
| Matriz de confusión | Sección 14 |
| Análisis visual de errores | Sección 15 |
| Overfitting y underfitting | Curvas y brecha en sección 12 |
| Test solo al final | Puerta explícita en sección 13 |
| Trazabilidad y reproducibilidad | Semillas, configuración, historiales y artefactos |
| Conclusiones | Sección 17, calculada con métricas reales |
