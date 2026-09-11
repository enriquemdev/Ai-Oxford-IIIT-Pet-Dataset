# Validación técnica de la solución

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

Las métricas de esa corrida no se incluyen porque no representan el experimento solicitado. La corrida final debe utilizar `FAST_MODE = False` y GPU en Google Colab.

## Dependencias verificadas

Se probó con TensorFlow 2.20, TensorFlow Datasets 4.9.9 y Keras 3. El archivo `requirements.txt` limita Protobuf a la rama 6 porque TFDS 4.9 todavía depende de una API retirada en Protobuf 7.

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
