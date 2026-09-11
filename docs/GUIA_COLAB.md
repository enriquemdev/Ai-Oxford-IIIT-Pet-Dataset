# Guía de ejecución final en Google Colab

## Resultado esperado

Una ejecución completa produce seis mejores checkpoints, historiales por época, tablas comparativas, matriz de confusión, métricas por clase, galería de errores, conclusiones basadas en los resultados y un archivo `entrega_oxford_pet.zip`.

## Preparación

1. Abra el notebook mediante el botón **Abrir en Google Colab** del README.
2. En Colab seleccione **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Compruebe en la primera salida que `Aceleradores` contiene un dispositivo GPU.
4. En la celda de configuración mantenga:

```python
FAST_MODE = False
RUN_TRAINING = True
FORCE_RETRAIN = False
```

5. Para no perder checkpoints si Colab desconecta la sesión, cambie:

```python
USE_GOOGLE_DRIVE = True
```

Colab solicitará autorización para montar Drive. Los resultados se guardarán en `MyDrive/oxford_pet_assignment`.

## Primera corrida

Ejecute **Entorno de ejecución → Ejecutar todas**. No interrumpa una época mientras se está escribiendo un checkpoint. Cada experimento guarda únicamente su mejor versión según `val_accuracy`.

El orden es deliberado:

1. Fully Connected.
2. CNN base.
3. CNN regularizada sin augmentation.
4. CNN regularizada con augmentation.
5. MobileNetV2 congelada.
6. Fine tuning de MobileNetV2.
7. Selección por validation.
8. Apertura y evaluación final de test.

## Si la sesión se desconecta

Vuelva a abrir el notebook, active GPU y Drive, deje `FORCE_RETRAIN = False` y ejecute todas las celdas. Los modelos con checkpoint se cargarán; solo se entrenará aquello que todavía no exista.

No cambie `FORCE_RETRAIN` a `True` salvo que quiera invalidar deliberadamente una corrida y reemplazar sus checkpoints.

## Revisión antes de entregar

Compruebe en las últimas celdas:

- cuál fue el modelo seleccionado por validation;
- si su test accuracy alcanza 85%;
- si el diagnóstico de curvas indica una brecha relevante;
- cuáles son las clases con menor precision, recall y F1;
- cuáles son los pares de razas más confundidos;
- si la ablación muestra beneficio o perjuicio del augmentation.

Si el modelo no alcanza 85%, el notebook lo indicará explícitamente. No edite el texto para afirmar un resultado distinto; conserve la evidencia y realice una nueva iteración usando únicamente train y validation.

## Archivos finales

La última celda crea `entrega_oxford_pet.zip`. Descárguelo y conserve también:

- el notebook ejecutado en `.ipynb`;
- una exportación PDF legible del notebook;
- el informe `artifacts/informe_resultados.md`;
- el checkpoint del modelo seleccionado;
- `run_config.json` para trazabilidad.

Al exportar el PDF, revise que ninguna figura o tabla quede cortada entre páginas.
