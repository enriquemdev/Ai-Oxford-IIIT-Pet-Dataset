"""Build the submitted notebook deterministically from readable cell sources."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Actividad_grupal_SCA.ipynb"


def lines(text: str) -> list[str]:
    value = dedent(text).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        r"""
        # Clasificación de razas con Oxford IIIT Pet

        ## Actividad grupal de Redes Neuronales Convolucionales

        Este notebook implementa un estudio experimental reproducible para clasificar las **37 razas** del Oxford-IIIT Pet Dataset. La entrega compara un baseline Fully Connected, arquitecturas CNN propias, una ablación controlada de data augmentation y dos etapas de transfer learning con MobileNetV2.

        **Objetivo verificable:** seleccionar el modelo únicamente con datos de validación y comprobar al final si supera **85% de accuracy en test** sin evidencia relevante de overfitting ni underfitting. El notebook no contiene métricas inventadas: todas las tablas, figuras y conclusiones se generan a partir de la ejecución real.
        """
    ),
    markdown(
        r"""
        ## Metodología y reglas del experimento

        1. Se fija una semilla global y se registra el entorno de ejecución.
        2. Se corrige el split suministrado: `train[:80]` seleccionaba 80 imágenes, no 80%. Se usa `train[:80%]`, `train[80%:90%]` y `train[90%:]`.
        3. Las imágenes se redimensionan a 160 por 160 y se normalizan al intervalo [0, 1].
        4. Train y validation se utilizan para aprender, ajustar y seleccionar. **Test permanece cerrado** hasta finalizar la selección.
        5. Todos los experimentos guardan el mejor checkpoint por `val_accuracy`, su historial y su configuración.
        6. El modelo final se juzga con accuracy, precision, recall, F1, matriz de confusión, curvas y análisis visual de errores.

        La comparación `cnn_regularized_no_aug` frente a `cnn_regularized_aug` mantiene igual arquitectura, optimizador y regularización. La única diferencia es el aumento de datos, por lo que funciona como una ablación interpretable.
        """
    ),
    markdown(
        r"""
        ## 1 Librerías y configuración

        En Google Colab las dependencias principales suelen estar instaladas. Si una importación falla, ejecute antes:

        ```python
        %pip install -q -r requirements.txt
        ```

        Si abrió el notebook directamente desde GitHub y `requirements.txt` no está en el entorno de Colab, use:

        ```python
        %pip install -q "tensorflow-datasets>=4.9,<5" "protobuf>=6.31,<7" "seaborn>=0.13,<1" "scikit-learn>=1.4,<2"
        ```

        Después de una instalación que cambie TensorFlow o Protobuf, reinicie el entorno de ejecución antes de continuar.
        """
    ),
    code(
        r"""
        import json
        import os
        import platform
        import random
        import shutil
        import sys
        import time
        import warnings
        from collections import OrderedDict
        from pathlib import Path
        from zipfile import ZIP_DEFLATED, ZipFile

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import sklearn
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from sklearn.metrics import classification_report, confusion_matrix
        from tensorflow import keras
        from tensorflow.keras import layers

        warnings.filterwarnings("ignore", category=UserWarning)
        sns.set_theme(style="whitegrid", context="notebook")
        print("TensorFlow:", tf.__version__)
        print("TensorFlow Datasets:", tfds.__version__)
        print("Python:", sys.version.split()[0])
        print("Aceleradores:", tf.config.list_physical_devices("GPU"))
        """
    ),
    code(
        r"""
        # Configuración central. FAST_MODE solo sirve para verificar el flujo.
        SEED = 42
        IMG_SIZE = 160
        BATCH_SIZE = 32
        FAST_MODE = False
        USE_GOOGLE_DRIVE = False
        FORCE_RETRAIN = False
        RUN_TRAINING = True

        random.seed(SEED)
        np.random.seed(SEED)
        tf.keras.utils.set_random_seed(SEED)

        try:
            tf.config.experimental.enable_op_determinism()
            DETERMINISM = True
        except Exception:
            DETERMINISM = False

        if USE_GOOGLE_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
            WORK_ROOT = Path("/content/drive/MyDrive/oxford_pet_assignment")
        else:
            WORK_ROOT = Path.cwd()

        MODEL_DIR = WORK_ROOT / "models"
        ARTIFACT_DIR = WORK_ROOT / "artifacts"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        CONFIG = {
            "seed": SEED,
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "fast_mode": FAST_MODE,
            "deterministic_ops": DETERMINISM,
            "tensorflow": tf.__version__,
            "tensorflow_datasets": tfds.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        }
        (ARTIFACT_DIR / "run_config.json").write_text(
            json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        CONFIG
        """
    ),
    markdown(
        r"""
        ## 2 Dataset y partición

        Oxford-IIIT Pet contiene imágenes RGB y etiquetas de raza. El split de trabajo conserva el esquema 80/10/10 de la plantilla. En modo completo se esperan aproximadamente 2,944 imágenes para train, 368 para validation y 368 para test.

        **Corrección importante:** `train[:80]` significa “las primeras 80 observaciones”. El símbolo `%` es indispensable para utilizar el 80%.
        """
    ),
    code(
        r"""
        if FAST_MODE:
            split_spec = ["train[:10%]", "train[80%:82%]", "train[90%:92%]"]
        else:
            split_spec = ["train[:80%]", "train[80%:90%]", "train[90%:]"]

        read_config = tfds.ReadConfig(shuffle_seed=SEED)
        (raw_train, raw_val, raw_test), dataset_info = tfds.load(
            "oxford_iiit_pet",
            split=split_spec,
            as_supervised=True,
            with_info=True,
            shuffle_files=True,
            read_config=read_config,
        )

        CLASS_NAMES = dataset_info.features["label"].names
        NUM_CLASSES = dataset_info.features["label"].num_classes

        def cardinality(dataset):
            value = tf.data.experimental.cardinality(dataset).numpy()
            return int(value) if value >= 0 else sum(1 for _ in dataset)

        split_sizes = {
            "train": cardinality(raw_train),
            "validation": cardinality(raw_val),
            "test": cardinality(raw_test),
        }
        assert NUM_CLASSES == 37, f"Se esperaban 37 clases y se encontraron {NUM_CLASSES}"
        assert all(size > 0 for size in split_sizes.values())
        print("Clases:", NUM_CLASSES)
        print("Tamaños:", split_sizes)
        print("Primeras clases:", CLASS_NAMES[:5])
        """
    ),
    code(
        r"""
        AUTOTUNE = tf.data.AUTOTUNE

        def preprocess_image(image, label):
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
            image = tf.cast(image, tf.float32) / 255.0
            # La interpolación con antialias puede producir diferencias de coma
            # flotante mínimas fuera del intervalo; se fuerza el contrato [0, 1].
            image = tf.clip_by_value(image, 0.0, 1.0)
            label = tf.cast(label, tf.int32)
            return image, label

        def prepare_dataset(dataset, training=False):
            dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
            if training:
                dataset = dataset.shuffle(
                    min(split_sizes["train"], 2_000),
                    seed=SEED,
                    reshuffle_each_iteration=True,
                )
            return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

        train_batches = prepare_dataset(raw_train, training=True)
        val_batches = prepare_dataset(raw_val)
        test_batches = prepare_dataset(raw_test)

        sample_images, sample_labels = next(iter(train_batches))
        assert sample_images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
        assert float(tf.reduce_min(sample_images)) >= 0.0
        assert float(tf.reduce_max(sample_images)) <= 1.0
        print("Batch de imágenes:", sample_images.shape)
        print("Batch de etiquetas:", sample_labels.shape)
        """
    ),
    markdown(
        r"""
        ## 3 Análisis exploratorio

        Se inspeccionan tamaño, balance por clase y ejemplos representativos. Las clases tienen nombres de razas; algunas son visualmente similares, de modo que una accuracy global debe complementarse con métricas por clase y análisis de confusiones.
        """
    ),
    code(
        r"""
        def label_counts(dataset):
            labels = np.fromiter((int(label.numpy()) for _, label in dataset), dtype=np.int32)
            counts = np.bincount(labels, minlength=NUM_CLASSES)
            return pd.DataFrame({"class_id": range(NUM_CLASSES), "class_name": CLASS_NAMES, "count": counts})

        train_distribution = label_counts(raw_train)
        val_distribution = label_counts(raw_val)
        test_distribution = label_counts(raw_test)

        distribution_summary = pd.DataFrame({
            "split": ["train", "validation", "test"],
            "images": [len(train_distribution) and train_distribution["count"].sum(),
                       len(val_distribution) and val_distribution["count"].sum(),
                       len(test_distribution) and test_distribution["count"].sum()],
            "min_per_class": [train_distribution["count"].min(), val_distribution["count"].min(), test_distribution["count"].min()],
            "max_per_class": [train_distribution["count"].max(), val_distribution["count"].max(), test_distribution["count"].max()],
        })
        display(distribution_summary)
        display(train_distribution.sort_values("count").head(10))
        train_distribution.to_csv(ARTIFACT_DIR / "class_distribution_train.csv", index=False)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        ordered = train_distribution.sort_values("count")
        axes[0].barh(ordered["class_name"], ordered["count"], color="#3b82f6")
        axes[0].set_title("Distribución de clases en train")
        axes[0].set_xlabel("Número de imágenes")

        preview_images, preview_labels = next(iter(prepare_dataset(raw_train.take(12))))
        axes[1].axis("off")
        axes[1].set_title("La galería se muestra en la figura siguiente")
        plt.tight_layout()
        fig.savefig(ARTIFACT_DIR / "eda_class_distribution.png", dpi=160, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(3, 4, figsize=(12, 10))
        for ax, image, label in zip(axes.flat, preview_images, preview_labels):
            ax.imshow(image)
            ax.set_title(CLASS_NAMES[int(label)])
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(ARTIFACT_DIR / "eda_examples.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        r"""
        ### Interpretación del EDA

        La tabla permite verificar de forma cuantitativa si el corte porcentual introduce desbalance relevante. Aunque el dataset completo contiene cerca de 200 imágenes por raza, este trabajo usa el split `train` suministrado y reserva parte de él para validación y test. La variación de pose, escala, iluminación, fondo y encuadre aumenta la dificultad; además, varias razas comparten color, textura y morfología.

        No se aplica oversampling antes de observar las métricas. Data augmentation se estudia mediante una comparación controlada, y las métricas macro evitan que una clase frecuente domine la conclusión.
        """
    ),
    markdown(
        r"""
        ## 4 Diseño experimental

        | Experimento | Arquitectura | Augmentation | Regularización | Propósito |
        |---|---|---|---|---|
        | `fc_baseline` | Flatten y Dense | No | Dropout | Baseline sin sesgo espacial |
        | `cnn_baseline` | 3 bloques Conv2D | No | No | Aporte básico de convoluciones |
        | `cnn_regularized_no_aug` | CNN propia profunda | No | BN y Dropout | Efecto de arquitectura y regularización |
        | `cnn_regularized_aug` | Misma CNN propia | Sí | BN y Dropout | Ablación del augmentation |
        | `mobilenet_transfer` | MobileNetV2 congelada | Sí | Dropout | Transfer learning |
        | `mobilenet_finetuned` | MobileNetV2 parcial | Sí | Dropout y LR bajo | Adaptación final al dominio |

        Todos usan sparse categorical cross-entropy y Adam. Las épocas máximas no son equivalentes a épocas efectivas porque EarlyStopping detiene el entrenamiento cuando validation deja de mejorar.
        """
    ),
    code(
        r"""
        DATA_AUGMENTATION = keras.Sequential(
            [
                layers.RandomFlip("horizontal", seed=SEED),
                layers.RandomRotation(0.08, fill_mode="reflect", seed=SEED + 1),
                layers.RandomZoom(0.12, fill_mode="reflect", seed=SEED + 2),
                layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=SEED + 3),
                layers.RandomContrast(0.10, seed=SEED + 4),
            ],
            name="data_augmentation",
        )

        def compile_model(model, learning_rate):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )
            return model

        def conv_block(x, filters, dropout=0.0):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
            if dropout:
                x = layers.SpatialDropout2D(dropout)(x)
            return x

        def build_fully_connected():
            inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
            x = layers.Resizing(64, 64)(inputs)
            x = layers.Flatten()(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.45)(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            return keras.Model(inputs, outputs, name="fc_baseline")

        def build_cnn_baseline():
            inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
            x = inputs
            for filters in (32, 64, 128):
                x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
                x = layers.MaxPooling2D()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation="relu")(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            return keras.Model(inputs, outputs, name="cnn_baseline")

        def build_regularized_cnn(use_augmentation):
            inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
            x = DATA_AUGMENTATION(inputs) if use_augmentation else inputs
            x = conv_block(x, 32, 0.05)
            x = conv_block(x, 64, 0.10)
            x = conv_block(x, 128, 0.15)
            x = conv_block(x, 256, 0.20)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(0.45)(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            suffix = "aug" if use_augmentation else "no_aug"
            return keras.Model(inputs, outputs, name=f"cnn_regularized_{suffix}")

        def build_mobilenet_transfer():
            inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
            x = DATA_AUGMENTATION(inputs)
            # El pipeline entrega [0,1]; MobileNetV2 fue entrenada con [-1,1].
            x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_normalization")(x)
            backbone = keras.applications.MobileNetV2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights="imagenet",
            )
            backbone.trainable = False
            x = backbone(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.35)(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            return keras.Model(inputs, outputs, name="mobilenet_transfer")
        """
    ),
    markdown(
        r"""
        ## 5 Utilidades de entrenamiento y trazabilidad

        Cada corrida persiste el mejor modelo completo, historial, log por época y metadatos. Si el checkpoint existe y `FORCE_RETRAIN` es falso, se carga en vez de entrenarse de nuevo. Esto permite retomar una sesión y realizar predicciones sin GPU.
        """
    ),
    code(
        r"""
        EXPERIMENT_CONFIG = OrderedDict({
            "fc_baseline": {"epochs": 18, "lr": 1e-3, "patience": 5},
            "cnn_baseline": {"epochs": 30, "lr": 1e-3, "patience": 6},
            "cnn_regularized_no_aug": {"epochs": 35, "lr": 7e-4, "patience": 7},
            "cnn_regularized_aug": {"epochs": 40, "lr": 7e-4, "patience": 8},
            "mobilenet_transfer": {"epochs": 20, "lr": 1e-3, "patience": 5},
            "mobilenet_finetuned": {"epochs": 20, "lr": 1e-5, "patience": 6},
        })

        if FAST_MODE:
            for config in EXPERIMENT_CONFIG.values():
                config["epochs"] = 2
                config["patience"] = 1

        models = OrderedDict()
        histories = OrderedDict()
        durations = OrderedDict()

        def experiment_paths(name):
            return {
                "model": MODEL_DIR / f"{name}.keras",
                "history": ARTIFACT_DIR / f"{name}_history.json",
                "log": ARTIFACT_DIR / f"{name}_epochs.csv",
            }

        def make_callbacks(name, config):
            paths = experiment_paths(name)
            return [
                keras.callbacks.ModelCheckpoint(
                    paths["model"], monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", mode="max", patience=config["patience"], restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", mode="min", factor=0.3, patience=max(2, config["patience"] // 2), min_lr=1e-7, verbose=1
                ),
                keras.callbacks.CSVLogger(paths["log"]),
                keras.callbacks.TerminateOnNaN(),
            ]

        def save_history(name, history_dict):
            serializable = {key: [float(value) for value in values] for key, values in history_dict.items()}
            experiment_paths(name)["history"].write_text(
                json.dumps(serializable, indent=2), encoding="utf-8"
            )
            return serializable

        def train_or_load(name, builder):
            config = EXPERIMENT_CONFIG[name]
            paths = experiment_paths(name)
            if paths["model"].exists() and not FORCE_RETRAIN:
                print(f"Cargando checkpoint existente: {paths['model']}")
                model = keras.models.load_model(paths["model"], compile=False)
                model = compile_model(model, config["lr"])
                history = json.loads(paths["history"].read_text()) if paths["history"].exists() else {}
                duration = np.nan
            else:
                if not RUN_TRAINING:
                    raise FileNotFoundError(f"No existe {paths['model']} y RUN_TRAINING=False")
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                model = compile_model(builder(), config["lr"])
                print(f"\nEntrenando {name} con {model.count_params():,} parámetros")
                started = time.perf_counter()
                fitted = model.fit(
                    train_batches,
                    validation_data=val_batches,
                    epochs=config["epochs"],
                    callbacks=make_callbacks(name, config),
                    verbose=2,
                )
                duration = time.perf_counter() - started
                history = save_history(name, fitted.history)
                model = keras.models.load_model(paths["model"], compile=False)
                model = compile_model(model, config["lr"])
            models[name] = model
            histories[name] = history
            durations[name] = duration
            return model

        EXPERIMENT_CONFIG
        """
    ),
    markdown(
        r"""
        ## 6 Experimento Fully Connected

        El baseline aplana una versión 64 por 64 para mantener un número razonable de parámetros. Al perder explícitamente la vecindad espacial, sirve para contrastar la inductive bias de las convoluciones.
        """
    ),
    code(
        r"""
        fc_model = train_or_load("fc_baseline", build_fully_connected)
        fc_model.summary()
        """
    ),
    markdown(
        r"""
        ## 7 Experimento CNN propia base

        Esta arquitectura introduce filtros locales y pooling, pero evita augmentation y regularización avanzada. Su resultado permite medir el salto respecto al modelo Fully Connected.
        """
    ),
    code(
        r"""
        cnn_baseline_model = train_or_load("cnn_baseline", build_cnn_baseline)
        cnn_baseline_model.summary()
        """
    ),
    markdown(
        r"""
        ## 8 CNN propia regularizada sin Data augmentation

        La red propuesta incorpora mayor profundidad, Batch Normalization, Spatial Dropout y Global Average Pooling. Esta primera variante no transforma imágenes y actúa como control de la ablación.
        """
    ),
    code(
        r"""
        cnn_regularized_no_aug_model = train_or_load(
            "cnn_regularized_no_aug", lambda: build_regularized_cnn(use_augmentation=False)
        )
        cnn_regularized_no_aug_model.summary()
        """
    ),
    markdown(
        r"""
        ## 9 CNN propia regularizada con Data augmentation

        Se reutiliza exactamente la misma CNN y se activan flip horizontal, rotación, zoom, traslación y contraste moderados. No se usan flips verticales porque no representan una variación natural de las mascotas.
        """
    ),
    code(
        r"""
        cnn_regularized_aug_model = train_or_load(
            "cnn_regularized_aug", lambda: build_regularized_cnn(use_augmentation=True)
        )
        cnn_regularized_aug_model.summary()
        """
    ),
    markdown(
        r"""
        ## 10 Transfer learning con MobileNetV2

        MobileNetV2 aporta representaciones aprendidas en ImageNet. En esta fase el backbone permanece congelado y solo se entrena la cabeza multiclase. Esto reduce el riesgo de destruir características útiles con un dataset relativamente pequeño.
        """
    ),
    code(
        r"""
        mobilenet_transfer_model = train_or_load("mobilenet_transfer", build_mobilenet_transfer)
        mobilenet_transfer_model.summary()
        """
    ),
    markdown(
        r"""
        ## 11 Fine tuning parcial

        El fine tuning parte del **mejor checkpoint** de transfer learning. Se descongelan solo las últimas 30 capas del backbone, se mantienen congeladas las capas Batch Normalization y se reduce el learning rate a `1e-5`. De este modo se adapta la representación a las razas sin modificar agresivamente los pesos preentrenados.
        """
    ),
    code(
        r"""
        def prepare_finetuned_model():
            transfer_path = experiment_paths("mobilenet_transfer")["model"]
            if not transfer_path.exists():
                raise FileNotFoundError("Primero debe existir el checkpoint de mobilenet_transfer")
            model = keras.models.load_model(transfer_path, compile=False)
            backbone = next(
                layer for layer in model.layers
                if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower()
            )
            backbone.trainable = True
            for layer in backbone.layers[:-30]:
                layer.trainable = False
            for layer in backbone.layers[-30:]:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
            model._name = "mobilenet_finetuned"
            return model

        mobilenet_finetuned_model = train_or_load("mobilenet_finetuned", prepare_finetuned_model)
        mobilenet_finetuned_model.summary()
        """
    ),
    markdown(
        r"""
        ## 12 Curvas y selección con validation

        La selección ocurre antes de cualquier evaluación en test. Se comparan los checkpoints mediante validation y se diagnostica la brecha entre train y validation en la época de mejor `val_accuracy`.
        """
    ),
    code(
        r"""
        def history_diagnostics(history):
            if not history or "val_accuracy" not in history:
                return {"best_epoch": np.nan, "train_accuracy": np.nan, "val_accuracy": np.nan,
                        "generalization_gap": np.nan, "diagnosis": "historial no disponible"}
            best_index = int(np.argmax(history["val_accuracy"]))
            train_acc = float(history["accuracy"][best_index])
            val_acc = float(history["val_accuracy"][best_index])
            gap = train_acc - val_acc
            if val_acc < 0.50:
                diagnosis = "posible underfitting"
            elif gap > 0.10:
                diagnosis = "posible overfitting"
            else:
                diagnosis = "sin evidencia fuerte en las curvas"
            return {
                "best_epoch": best_index + 1,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "generalization_gap": gap,
                "diagnosis": diagnosis,
            }

        validation_rows = []
        for name, model in models.items():
            val_loss, val_accuracy = model.evaluate(val_batches, verbose=0)
            diagnostics = history_diagnostics(histories[name])
            validation_rows.append({
                "model": name,
                "parameters": model.count_params(),
                "best_epoch": diagnostics["best_epoch"],
                "train_accuracy_at_best": diagnostics["train_accuracy"],
                "val_accuracy_history": diagnostics["val_accuracy"],
                "val_accuracy_checkpoint": float(val_accuracy),
                "generalization_gap": diagnostics["generalization_gap"],
                "curve_diagnosis": diagnostics["diagnosis"],
                "training_seconds": durations[name],
            })

        validation_results = pd.DataFrame(validation_rows).sort_values(
            "val_accuracy_checkpoint", ascending=False
        ).reset_index(drop=True)
        validation_results.to_csv(ARTIFACT_DIR / "validation_model_comparison.csv", index=False)
        display(validation_results.style.format({
            "train_accuracy_at_best": "{:.3f}", "val_accuracy_history": "{:.3f}",
            "val_accuracy_checkpoint": "{:.3f}", "generalization_gap": "{:.3f}",
            "training_seconds": "{:.1f}"
        }))

        best_validation_model = validation_results.iloc[0]["model"]
        print("Modelo seleccionado exclusivamente con validation:", best_validation_model)
        """
    ),
    code(
        r"""
        def plot_histories(histories_to_plot, filename):
            available = {name: h for name, h in histories_to_plot.items() if h and "accuracy" in h}
            if not available:
                print("No hay historiales guardados para graficar.")
                return
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for name, history in available.items():
                epochs = np.arange(1, len(history["accuracy"]) + 1)
                axes[0].plot(epochs, history["accuracy"], alpha=0.75, label=f"{name} train")
                axes[0].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{name} val")
                axes[1].plot(epochs, history["loss"], alpha=0.75, label=f"{name} train")
                axes[1].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
            axes[0].set(title="Accuracy por época", xlabel="Época", ylabel="Accuracy")
            axes[1].set(title="Loss por época", xlabel="Época", ylabel="Loss")
            for ax in axes:
                ax.legend(fontsize=7)
                ax.grid(alpha=0.25)
            plt.tight_layout()
            fig.savefig(ARTIFACT_DIR / filename, dpi=170, bbox_inches="tight")
            plt.show()

        plot_histories(histories, "all_training_curves.png")
        plot_histories(
            {best_validation_model: histories[best_validation_model]},
            "best_model_training_curves.png",
        )
        """
    ),
    markdown(
        r"""
        ## 13 Evaluación final en test

        A partir de este punto el diseño y la selección están cerrados. Test se abre una sola vez para obtener la comparación final. Sus métricas no deben utilizarse para volver atrás y cambiar hiperparámetros.
        """
    ),
    code(
        r"""
        # FINAL TEST GATE: no mover esta evaluación antes de best_validation_model.
        y_true = np.concatenate([labels.numpy() for _, labels in test_batches])
        test_probabilities = {}
        test_rows = []

        for name, model in models.items():
            probabilities = model.predict(test_batches, verbose=0)
            predictions = probabilities.argmax(axis=1)
            report = classification_report(
                y_true,
                predictions,
                labels=np.arange(NUM_CLASSES),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
            loss, accuracy = model.evaluate(test_batches, verbose=0)
            test_probabilities[name] = probabilities
            test_rows.append({
                "model": name,
                "selected_by_validation": name == best_validation_model,
                "test_loss": float(loss),
                "test_accuracy": float(accuracy),
                "macro_precision": float(report["macro avg"]["precision"]),
                "macro_recall": float(report["macro avg"]["recall"]),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]),
            })

        test_results = pd.DataFrame(test_rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
        test_results.to_csv(ARTIFACT_DIR / "final_test_model_comparison.csv", index=False)
        display(test_results.style.format({
            "test_loss": "{:.4f}", "test_accuracy": "{:.3%}", "macro_precision": "{:.3%}",
            "macro_recall": "{:.3%}", "macro_f1": "{:.3%}", "weighted_f1": "{:.3%}"
        }))

        selected_test_row = test_results[test_results["model"] == best_validation_model].iloc[0]
        print(f"Accuracy final del modelo seleccionado: {selected_test_row['test_accuracy']:.2%}")
        print("Objetivo de 85%:", "CUMPLIDO" if selected_test_row["test_accuracy"] >= 0.85 else "NO CUMPLIDO")
        """
    ),
    code(
        r"""
        fig, ax = plt.subplots(figsize=(11, 5))
        order = test_results.sort_values("test_accuracy")
        colors = ["#16a34a" if selected else "#64748b" for selected in order["selected_by_validation"]]
        ax.barh(order["model"], order["test_accuracy"], color=colors)
        ax.axvline(0.85, color="#dc2626", linestyle="--", label="Objetivo 85%")
        ax.set(xlabel="Test accuracy", title="Comparación final sin selección sobre test", xlim=(0, 1))
        ax.legend()
        for index, value in enumerate(order["test_accuracy"]):
            ax.text(min(value + 0.01, 0.96), index, f"{value:.1%}", va="center")
        plt.tight_layout()
        fig.savefig(ARTIFACT_DIR / "final_test_accuracy.png", dpi=170, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 14 Precision recall y matriz de confusión

        El análisis detallado se realiza sobre el modelo elegido por validation, incluso si otro modelo obtiene accidentalmente mejor test. Esto conserva la independencia metodológica del conjunto final.
        """
    ),
    code(
        r"""
        selected_probabilities = test_probabilities[best_validation_model]
        selected_predictions = selected_probabilities.argmax(axis=1)
        selected_report_dict = classification_report(
            y_true,
            selected_predictions,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        per_class_report = pd.DataFrame(selected_report_dict).T.loc[CLASS_NAMES]
        per_class_report.index.name = "class_name"
        per_class_report = per_class_report.sort_values("f1-score")
        per_class_report.to_csv(ARTIFACT_DIR / "selected_model_per_class_metrics.csv")

        print("Diez clases con menor F1")
        display(per_class_report.head(10).style.format("{:.3f}"))
        print("Diez clases con mayor F1")
        display(per_class_report.tail(10).sort_values("f1-score", ascending=False).style.format("{:.3f}"))
        """
    ),
    code(
        r"""
        cm = confusion_matrix(y_true, selected_predictions, labels=np.arange(NUM_CLASSES))
        cm_normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        np.save(ARTIFACT_DIR / "selected_model_confusion_matrix.npy", cm)

        fig, ax = plt.subplots(figsize=(18, 15))
        sns.heatmap(
            cm_normalized,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={"label": "Proporción por clase real"},
            ax=ax,
        )
        ax.set(title=f"Matriz de confusión normalizada — {best_validation_model}",
               xlabel="Clase predicha", ylabel="Clase real")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        fig.savefig(ARTIFACT_DIR / "selected_model_confusion_matrix.png", dpi=180, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        r"""
        confusion_rows = []
        for true_id in range(NUM_CLASSES):
            for predicted_id in range(NUM_CLASSES):
                if true_id != predicted_id and cm[true_id, predicted_id] > 0:
                    confusion_rows.append({
                        "real": CLASS_NAMES[true_id],
                        "predicted": CLASS_NAMES[predicted_id],
                        "errors": int(cm[true_id, predicted_id]),
                        "rate_within_real_class": float(cm_normalized[true_id, predicted_id]),
                    })
        confusion_pairs = pd.DataFrame(confusion_rows).sort_values(
            ["errors", "rate_within_real_class"], ascending=False
        ).reset_index(drop=True)
        confusion_pairs.to_csv(ARTIFACT_DIR / "selected_model_confusion_pairs.csv", index=False)
        print("Pares de confusión más frecuentes")
        display(confusion_pairs.head(15).style.format({"rate_within_real_class": "{:.1%}"}))
        """
    ),
    markdown(
        r"""
        ## 15 Análisis de errores

        La galería ordena errores por confianza. Los errores de alta confianza son especialmente informativos: pueden señalar similitud morfológica entre razas, fondos dominantes, encuadres extremos, o ejemplos atípicos. Este análisis no se usa para reajustar el modelo después de abrir test.
        """
    ),
    code(
        r"""
        test_images = np.concatenate([images.numpy() for images, _ in test_batches])
        prediction_confidence = selected_probabilities.max(axis=1)
        wrong_indices = np.flatnonzero(selected_predictions != y_true)
        wrong_indices = wrong_indices[np.argsort(prediction_confidence[wrong_indices])[::-1]]

        gallery_count = min(16, len(wrong_indices))
        if gallery_count:
            rows = int(np.ceil(gallery_count / 4))
            fig, axes = plt.subplots(rows, 4, figsize=(14, 3.6 * rows))
            axes = np.atleast_1d(axes).ravel()
            for ax in axes:
                ax.axis("off")
            for ax, index in zip(axes, wrong_indices[:gallery_count]):
                ax.imshow(test_images[index])
                ax.set_title(
                    f"Real: {CLASS_NAMES[y_true[index]]}\n"
                    f"Pred: {CLASS_NAMES[selected_predictions[index]]}\n"
                    f"Confianza: {prediction_confidence[index]:.1%}",
                    fontsize=9,
                )
                ax.axis("off")
            plt.suptitle(f"Errores de mayor confianza — {best_validation_model}", y=1.01)
            plt.tight_layout()
            fig.savefig(ARTIFACT_DIR / "selected_model_error_gallery.png", dpi=170, bbox_inches="tight")
            plt.show()
        else:
            print("No hubo errores en test.")
        """
    ),
    markdown(
        r"""
        ## 16 Impacto de Data augmentation

        La siguiente comparación aísla augmentation porque ambos modelos comparten arquitectura e hiperparámetros. Una mejora en validation/test junto con una brecha de generalización menor apoya que las transformaciones ayudan. Si empeora, debe discutirse si la intensidad elegida distorsiona señales finas de la raza.
        """
    ),
    code(
        r"""
        ablation_names = ["cnn_regularized_no_aug", "cnn_regularized_aug"]
        ablation = validation_results[validation_results["model"].isin(ablation_names)].merge(
            test_results[["model", "test_accuracy", "macro_f1"]], on="model", how="left"
        ).set_index("model").loc[ablation_names]
        display(ablation.style.format({
            "train_accuracy_at_best": "{:.3f}", "val_accuracy_history": "{:.3f}",
            "val_accuracy_checkpoint": "{:.3f}", "generalization_gap": "{:.3f}",
            "test_accuracy": "{:.3f}", "macro_f1": "{:.3f}", "training_seconds": "{:.1f}"
        }))

        augmentation_delta = (
            ablation.loc["cnn_regularized_aug", "test_accuracy"]
            - ablation.loc["cnn_regularized_no_aug", "test_accuracy"]
        )
        direction = "mejoró" if augmentation_delta > 0 else "redujo"
        print(f"Data augmentation {direction} el test accuracy en {abs(augmentation_delta):.2%} puntos porcentuales.")
        """
    ),
    markdown(
        r"""
        ## 17 Conclusiones

        Las conclusiones siguientes se construyen con resultados observados. El criterio académico se considera cumplido solo si el modelo seleccionado por validation supera 85% en test y sus curvas no muestran una brecha de generalización relevante. El diagnóstico automático es una ayuda; la interpretación final debe considerar conjuntamente curvas, métricas por clase y errores visuales.
        """
    ),
    code(
        r"""
        selected_validation = validation_results[
            validation_results["model"] == best_validation_model
        ].iloc[0]
        selected_test = test_results[test_results["model"] == best_validation_model].iloc[0]

        objective_met = selected_test["test_accuracy"] >= 0.85
        curve_ok = selected_validation["curve_diagnosis"] == "sin evidencia fuerte en las curvas"
        fc_accuracy = float(test_results.loc[test_results["model"] == "fc_baseline", "test_accuracy"].iloc[0])
        cnn_gain = float(selected_test["test_accuracy"] - fc_accuracy)

        print("CONCLUSIÓN BASADA EN LA EJECUCIÓN")
        print(f"1. El modelo seleccionado por validation fue {best_validation_model}.")
        print(f"2. Alcanzó {selected_test['test_accuracy']:.2%} de accuracy y {selected_test['macro_f1']:.2%} de macro-F1 en test.")
        print(f"3. El umbral académico de 85% {'se cumplió' if objective_met else 'no se cumplió'}.")
        print(f"4. El diagnóstico de curvas fue: {selected_validation['curve_diagnosis']}.")
        print(f"5. Frente al Fully Connected, el modelo seleccionado cambió el accuracy en {cnn_gain:+.2%} puntos porcentuales.")
        print(f"6. La ablación midió un efecto de augmentation de {augmentation_delta:+.2%} puntos porcentuales en test.")
        print("7. Las clases y pares concretos que requieren mayor atención aparecen en las tablas anteriores.")

        if objective_met and curve_ok:
            print("\nVeredicto: el objetivo se cumple sin evidencia fuerte de overfitting o underfitting en las curvas.")
        elif objective_met:
            print("\nVeredicto: se supera 85%, pero las curvas requieren una discusión crítica antes de afirmar buena generalización.")
        else:
            print("\nVeredicto: la ejecución aún no satisface el umbral; no debe reportarse como cumplido.")
        """
    ),
    markdown(
        r"""
        ## 18 Exportación de resultados y entrega

        Esta celda genera un informe Markdown con las métricas reales y empaqueta checkpoints, historiales, tablas y figuras. El notebook ejecutado puede exportarse a PDF desde el navegador para conservar la narrativa completa y las salidas visibles.
        """
    ),
    code(
        r"""
        def markdown_table(frame, columns):
            header = "| " + " | ".join(columns) + " |"
            separator = "|" + "|".join(["---"] * len(columns)) + "|"
            rows = []
            for _, row in frame[columns].iterrows():
                values = []
                for column in columns:
                    value = row[column]
                    if isinstance(value, (float, np.floating)):
                        values.append(f"{value:.4f}")
                    else:
                        values.append(str(value))
                rows.append("| " + " | ".join(values) + " |")
            return "\n".join([header, separator, *rows])

        worst_classes = per_class_report.reset_index().head(10)
        report_text = f'''# Informe de resultados Oxford IIIT Pet

        ## Resultado principal

        El modelo seleccionado exclusivamente con validation fue **{best_validation_model}**. En test obtuvo **{selected_test['test_accuracy']:.2%} de accuracy** y **{selected_test['macro_f1']:.2%} de macro F1**. El umbral de 85% **{'se cumplió' if objective_met else 'no se cumplió'}**. El diagnóstico de curvas fue **{selected_validation['curve_diagnosis']}**.

        ## Comparación final

        {markdown_table(test_results, ['model', 'selected_by_validation', 'test_accuracy', 'macro_precision', 'macro_recall', 'macro_f1'])}

        ## Ablación de augmentation

        La diferencia controlada en test accuracy fue {augmentation_delta:+.2%} puntos porcentuales.

        ## Clases con menor F1

        {markdown_table(worst_classes, ['class_name', 'precision', 'recall', 'f1-score', 'support'])}

        ## Pares de confusión principales

        {markdown_table(confusion_pairs.head(10), ['real', 'predicted', 'errors', 'rate_within_real_class'])}

        ## Trazabilidad

        La configuración completa está en `run_config.json`; los historiales y checkpoints permiten reproducir las figuras y predicciones sin reentrenar. Test se abrió después de seleccionar el modelo con validation.
        '''
        report_text = "\n".join(line.strip() for line in report_text.splitlines()) + "\n"
        (ARTIFACT_DIR / "informe_resultados.md").write_text(report_text, encoding="utf-8")

        archive_path = Path.cwd() / "entrega_oxford_pet.zip"
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for folder in (MODEL_DIR, ARTIFACT_DIR):
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.write(path, arcname=f"{folder.name}/{path.relative_to(folder)}")

        print("Informe:", ARTIFACT_DIR / "informe_resultados.md")
        print("Paquete de entrega:", archive_path)

        try:
            from google.colab import files
            print("En Colab puede descargar el paquete con: files.download(str(archive_path))")
        except ImportError:
            pass
        """
    ),
    markdown(
        r"""
        ## Referencias

        - Oxford Visual Geometry Group. *The Oxford-IIIT Pet Dataset*.
        - TensorFlow Datasets. *Oxford IIIT Pet catalog*.
        - TensorFlow. *Transfer learning and fine tuning*.
        - Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.

        Los enlaces originales del enunciado y el material suministrado se conservan en la carpeta `materiales` del repositorio.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Actividad_grupal_SCA.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
