"""Generate the final academic PDF from observed notebook artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "output" / "pdf" / "informe_tecnico_oxford_pet.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#1F6F8B")
PALE_BLUE = colors.HexColor("#EAF3F7")
LIGHT = colors.HexColor("#F4F6F8")
GRID = colors.HexColor("#D9E0E6")
TEXT = colors.HexColor("#202A33")
MUTED = colors.HexColor("#586875")
GREEN = colors.HexColor("#238B57")


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pct(value: str | float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def image(path: Path, width: float, height: float) -> Image:
    item = Image(str(path))
    item._restrictSize(width, height)
    return item


def table(data, widths, *, header=True, font_size=8.0, row_bgs=True) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]
    if row_bgs:
        first = 1 if header else 0
        for row in range(first, len(data)):
            if (row - first) % 2:
                commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    t.setStyle(TableStyle(commands))
    return t


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: Path):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=1.7 * cm,
            rightMargin=1.7 * cm,
            topMargin=1.7 * cm,
            bottomMargin=1.65 * cm,
            title="Clasificacion de razas con Oxford IIIT Pet",
            author="Actividad grupal de Sistemas Cognitivos Artificiales",
            subject="Informe tecnico de redes neuronales convolucionales",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 0.82 * cm, "Oxford IIIT Pet - Actividad grupal")
        canvas.drawRightString(A4[0] - doc.rightMargin, 0.82 * cm, f"Pagina {doc.page}")
        canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    validation = read_csv("validation_model_comparison.csv")
    tests = read_csv("final_test_model_comparison.csv")
    per_class = read_csv("selected_model_per_class_metrics.csv")
    confusions = read_csv("selected_model_confusion_pairs.csv")

    tests_by_model = {row["model"]: row for row in tests}
    validation_by_model = {row["model"]: row for row in validation}
    selected = next(row for row in tests if row["selected_by_validation"] == "True")
    selected_name = selected["model"]
    selected_validation = validation_by_model[selected_name]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25,
        leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "CoverSub", parent=styles["Normal"], fontName="Helvetica", fontSize=13,
        leading=19, textColor=BLUE, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16,
        leading=20, textColor=NAVY, spaceBefore=5, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        "H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        "Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4,
        leading=13.2, textColor=TEXT, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        "Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8,
        leading=10.2, textColor=MUTED, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Captionx", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.8,
        leading=10, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        "Metric", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=19,
        leading=22, textColor=GREEN, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "MetricLabel", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.5,
        leading=9, textColor=MUTED, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.5,
        leading=9, textColor=colors.white, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6,
        leading=9.2, textColor=TEXT,
    ))

    body = styles["Bodyx"]
    h1 = styles["H1x"]
    h2 = styles["H2x"]
    caption = styles["Captionx"]
    story = []

    # Cover and executive result
    story += [
        Spacer(1, 1.0 * cm),
        para("Clasificacion de razas con Oxford IIIT Pet", styles["CoverTitle"]),
        para("Actividad grupal de Redes Neuronales Convolucionales", styles["CoverSub"]),
        Spacer(1, 0.55 * cm),
        para(
            "Informe tecnico de la comparacion experimental entre un modelo Fully Connected, "
            "tres CNN propias y dos etapas de transferencia con MobileNetV2. La seleccion se "
            "realizo exclusivamente con validacion y el conjunto de test se abrio al final.",
            body,
        ),
        Spacer(1, 0.35 * cm),
    ]
    metric_data = [
        [para(pct(selected["test_accuracy"], 2), styles["Metric"]),
         para(pct(selected["macro_f1"], 2), styles["Metric"]),
         para(pct(selected_validation["val_accuracy_checkpoint"], 2), styles["Metric"])],
        [para("Accuracy final en test", styles["MetricLabel"]),
         para("Macro F1 en test", styles["MetricLabel"]),
         para("Accuracy de validacion", styles["MetricLabel"])],
    ]
    metrics_table = Table(metric_data, colWidths=[5.15 * cm] * 3, rowHeights=[1.0 * cm, 0.72 * cm])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [
        metrics_table,
        Spacer(1, 0.55 * cm),
        para(
            f"<b>Resultado principal.</b> <i>{selected_name}</i> alcanzo {pct(selected['test_accuracy'], 2)} "
            f"de accuracy y {pct(selected['macro_f1'], 2)} de macro F1. Supero el objetivo de 85% "
            f"con una brecha train-validation de {pct(selected_validation['generalization_gap'], 2)} "
            "en la mejor epoca, sin evidencia fuerte de overfitting o underfitting.",
            body,
        ),
        Spacer(1, 0.35 * cm),
        table([
            [para("Entorno", styles["TableHeader"]), para("Configuracion verificada", styles["TableHeader"])],
            ["Equipo", "Apple M5, 16 GB de memoria"],
            ["Aceleracion", "TensorFlow Metal - GPU detectada"],
            ["Software", "Python 3.12.14, TensorFlow 2.18.1, TFDS 4.9.10"],
            ["Datos", "2,944 train / 368 validation / 368 test; 37 razas"],
            ["Fecha de ejecucion", "11 de septiembre de 2026"],
        ], [4.1 * cm, 11.35 * cm], font_size=8.4),
        Spacer(1, 0.8 * cm),
        para("Sistemas Cognitivos Artificiales", styles["CoverSub"]),
        PageBreak(),
    ]

    # Objective and methodology
    story += [
        para("1 Objetivo y metodologia", h1),
        para(
            "El objetivo fue entrenar un clasificador de las 37 razas de perros y gatos del "
            "Oxford-IIIT Pet Dataset, lograr al menos 85% de accuracy en test y demostrar que el "
            "modelo seleccionado no presenta senales relevantes de sobreajuste o subajuste.", body),
        para("Protocolo experimental", h2),
        para(
            "La plantilla original contenia <font name='Courier'>train[:80]</font>, expresion que "
            "selecciona 80 observaciones. Se corrigio a un corte porcentual 80/10/10. Las imagenes "
            "se redimensionaron a 160 x 160, se normalizaron a [0, 1] y se procesaron con "
            "<font name='Courier'>tf.data</font>. Se fijaron semillas, se guardo el mejor checkpoint "
            "por accuracy de validacion y se aplico parada temprana.", body),
        table([
            ["Etapa", "Uso autorizado", "Control metodologico"],
            ["Train", "Ajuste de pesos", "Shuffle, batches, augmentation segun experimento"],
            ["Validation", "Seleccion y callbacks", "Mejor checkpoint por val_accuracy"],
            ["Test", "Evaluacion final", "No se uso para elegir modelo ni hiperparametros"],
        ], [3.1 * cm, 4.4 * cm, 7.95 * cm], font_size=8.2),
        para("Criterios de evaluacion", h2),
        para(
            "Se compararon accuracy, precision, recall, macro F1 y weighted F1. Las curvas de "
            "train y validation se analizaron en la epoca del mejor checkpoint. Una brecha mayor "
            "a 10 puntos se trato como posible overfitting y una accuracy de validacion menor de "
            "50% como posible underfitting. Este diagnostico automatico se complemento con la "
            "matriz de confusion y la revision visual de errores.", body),
        para("2 Analisis exploratorio", h1),
        para(
            "El split completo contiene 3,680 imagenes del conjunto train suministrado por TFDS. "
            "Cada particion conserva las 37 etiquetas, aunque el numero de ejemplos por clase en "
            "validation y test es reducido. El problema incluye variaciones de escala, pose, "
            "iluminacion, fondo y encuadre, ademas de razas con morfologia muy similar.", body),
        image(ARTIFACTS / "eda_examples.png", 15.8 * cm, 10.2 * cm),
        para("Figura 1. Muestra reproducible de imagenes del conjunto de entrenamiento.", caption),
        PageBreak(),
    ]

    # Experiment design
    experiment_rows = [
        ["Modelo", "Arquitectura", "Aug.", "Regularizacion", "Funcion experimental"],
        ["fc_baseline", "Resize 64, Flatten, Dense", "No", "Dropout", "Baseline sin estructura espacial"],
        ["cnn_baseline", "3 bloques Conv2D", "No", "Ninguna avanzada", "Aporte basico convolucional"],
        ["cnn_regularized_no_aug", "4 bloques CNN propios", "No", "BN + spatial dropout", "Control de la ablacion"],
        ["cnn_regularized_aug", "Misma CNN propia", "Si", "BN + spatial dropout", "Efecto aislado de augmentation"],
        ["mobilenet_transfer", "MobileNetV2 congelada", "Si", "Dropout", "Transfer learning"],
        ["mobilenet_finetuned", "Ultimas 30 capas habilitadas", "Si", "BN congelada + LR bajo", "Adaptacion final al dominio"],
    ]
    story += [
        para("3 Diseno experimental", h1),
        para(
            "Se ejecutaron seis experimentos. Las dos variantes de la CNN regularizada comparten "
            "arquitectura, optimizador e hiperparametros; la unica diferencia controlada es el "
            "aumento de datos. MobileNetV2 utiliza pesos de ImageNet y una normalizacion interna "
            "de [0, 1] a [-1, 1].", body),
        table(experiment_rows, [3.25 * cm, 3.8 * cm, 1.15 * cm, 3.35 * cm, 3.9 * cm], font_size=7.1),
        Spacer(1, 0.35 * cm),
        para("Trazabilidad y reproducibilidad", h2),
        para(
            "Cada experimento produjo un modelo Keras completo, historial JSON y registro CSV por "
            "epoca. Los checkpoints existentes se cargan sin reentrenar cuando "
            "<font name='Courier'>FORCE_RETRAIN=False</font>. La ejecucion final genero seis modelos "
            "y 27 artefactos; el entorno Metal se verifico antes del entrenamiento.", body),
        para("4 Resultados comparativos", h1),
    ]
    comparison = [["Modelo", "Val. acc.", "Test acc.", "Macro F1", "Diagnostico"]]
    order = [row["model"] for row in validation]
    for name in order:
        v = validation_by_model[name]
        t = tests_by_model[name]
        comparison.append([
            name,
            pct(v["val_accuracy_checkpoint"]),
            pct(t["test_accuracy"]),
            pct(t["macro_f1"]),
            v["curve_diagnosis"],
        ])
    story += [
        table(comparison, [4.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 5.25 * cm], font_size=7.7),
        Spacer(1, 0.25 * cm),
        image(ARTIFACTS / "final_test_accuracy.png", 15.8 * cm, 7.5 * cm),
        para("Figura 2. Accuracy final de los seis modelos; la linea roja marca el objetivo.", caption),
        para(
            "La transferencia produjo el cambio decisivo: MobileNetV2 congelada obtuvo 84.24% y "
            "el fine tuning parcial elevo el resultado a 86.14%, una mejora de 1.90 puntos. Las "
            "arquitecturas entrenadas desde cero permanecieron subajustadas bajo el presupuesto "
            "experimental, por lo que no compitieron con las representaciones preentrenadas.", body),
    ]

    # Curves and selected model
    story += [
        para("5 Seleccion y generalizacion", h1),
        para(
            f"El modelo se selecciono antes de abrir test. <i>{selected_name}</i> obtuvo "
            f"{pct(selected_validation['val_accuracy_checkpoint'], 2)} en validation y fue el mejor "
            "checkpoint. En su mejor epoca registro 91.03% en train y 88.04% en validation, una "
            "brecha de 2.99 puntos. La parada temprana restauro ese checkpoint aunque el entrenamiento "
            "continuara hasta completar la paciencia configurada.", body),
        image(ARTIFACTS / "best_model_training_curves.png", 16.0 * cm, 6.4 * cm),
        para("Figura 3. Curvas del modelo seleccionado; el mejor checkpoint corresponde a la epoca 2.", caption),
        para("Lectura de las curvas", h2),
        para(
            "La accuracy de train aumento despues de la mejor epoca, mientras validation se mantuvo "
            "entre aproximadamente 86.96% y 88.04%. La loss de validation no mostro una mejora "
            "sostenida posterior. Esta separacion moderada justifica conservar el checkpoint temprano; "
            "no alcanza el umbral de 10 puntos establecido para una alerta fuerte de overfitting.", body),
        para("Metricas finales del modelo seleccionado", h2),
        table([
            ["Accuracy", "Precision macro", "Recall macro", "F1 macro", "F1 ponderado"],
            [pct(selected["test_accuracy"], 2), pct(selected["macro_precision"], 2),
             pct(selected["macro_recall"], 2), pct(selected["macro_f1"], 2),
             pct(selected["weighted_f1"], 2)],
        ], [3.09 * cm] * 5, font_size=8.3),
        Spacer(1, 0.3 * cm),
        para(
            "La proximidad entre macro F1 (85.10%) y weighted F1 (86.01%) indica que el resultado "
            "global no depende exclusivamente de las clases con mas ejemplos. Sin embargo, el soporte "
            "por raza en test es pequeno; por ello las diferencias por clase deben interpretarse como "
            "evidencia descriptiva de esta particion, no como estimaciones definitivas.", body),
        PageBreak(),
    ]

    # Per-class metrics split deliberately across two pages.
    sorted_classes = sorted(per_class, key=lambda row: float(row["f1-score"]))
    def class_table(rows):
        data = [["Raza", "Precision", "Recall", "F1", "Soporte"]]
        for row in rows:
            data.append([row["class_name"], pct(row["precision"]), pct(row["recall"]),
                         pct(row["f1-score"]), str(int(float(row["support"])))])
        return table(data, [6.0 * cm, 2.35 * cm, 2.35 * cm, 2.35 * cm, 2.35 * cm], font_size=7.8)

    story += [
        para("6 Resultados por clase", h1),
        para(
            "Las 37 clases se ordenan de menor a mayor F1 para hacer visibles las debilidades. Beagle "
            "fue la clase mas dificil (62.50% F1). Ragdoll y Bengal obtuvieron 66.67%; Birman alcanzo "
            "71.43%. Varias razas lograron F1 perfecto en esta particion, pero con soportes de entre "
            "5 y 16 imagenes.", body),
        class_table(sorted_classes[:19]),
        para("Tabla 1. Metricas por clase, primera mitad, ordenadas por F1 ascendente.", caption),
        PageBreak(),
        para("6 Resultados por clase continuacion", h1),
        class_table(sorted_classes[19:]),
        para("Tabla 1. Metricas por clase, segunda mitad.", caption),
        para("Interpretacion", h2),
        para(
            "Los errores se concentran en razas visualmente proximas. Las metricas mas bajas no "
            "sugieren una sola falla sistematica: aparecen gatos de pelo largo, gatos con patrones "
            "similares y perros con siluetas compartidas. El pequeno soporte hace que dos o tres "
            "errores modifiquen de forma importante el recall de una clase.", body),
        PageBreak(),
    ]

    # Confusion matrix
    confusion_data = [["Raza real", "Prediccion", "Errores", "Tasa dentro de la clase"]]
    for row in confusions[:10]:
        confusion_data.append([row["real"], row["predicted"], row["errors"], pct(row["rate_within_real_class"])])
    story += [
        para("7 Matriz de confusion", h1),
        para(
            "La diagonal dominante confirma el buen rendimiento general. Los pares fuera de la diagonal "
            "muestran errores localizados, principalmente entre razas que comparten color, longitud de "
            "pelaje, forma facial o tamano corporal.", body),
        image(ARTIFACTS / "selected_model_confusion_matrix.png", 15.6 * cm, 13.1 * cm),
        para("Figura 4. Matriz de confusion normalizada del modelo MobileNetV2 con fine tuning.", caption),
        PageBreak(),
        para("7 Pares de confusion principales", h1),
        table(confusion_data, [4.5 * cm, 4.8 * cm, 2.2 * cm, 3.95 * cm], font_size=8.0),
        Spacer(1, 0.35 * cm),
        para(
            "Persian se confundio con Maine Coon en 3 de 7 casos; Bengal con Abyssinian en 3 de 8; "
            "y Birman con Ragdoll en 3 de 16. Estos pares son coherentes con similitudes morfologicas. "
            "Samoyed y Great Pyrenees comparten pelaje blanco abundante, mientras Birman, Ragdoll y "
            "Siamese comparten patrones de color point.", body),
        PageBreak(),
        para("8 Analisis visual de errores", h1),
        para(
            "La galeria prioriza errores de alta confianza. Los casos mas seguros incluyen Persian "
            "predicho como Maine Coon, Samoyed como Great Pyrenees y Birman como Ragdoll. Tambien "
            "aparecen sujetos pequenos, fondos dominantes, iluminacion atipica y encuadres donde la "
            "morfologia completa no es visible.", body),
        image(ARTIFACTS / "selected_model_error_gallery.png", 15.6 * cm, 12.2 * cm),
        para("Figura 5. Dieciseis errores con mayor confianza del modelo seleccionado.", caption),
        PageBreak(),
    ]

    # Augmentation, conclusions, references
    no_aug = tests_by_model["cnn_regularized_no_aug"]
    aug = tests_by_model["cnn_regularized_aug"]
    delta = float(aug["test_accuracy"]) - float(no_aug["test_accuracy"])
    story += [
        para("9 Impacto de data augmentation", h1),
        para(
            "La ablacion controlada no mostro una mejora. La CNN regularizada sin augmentation alcanzo "
            f"{pct(no_aug['test_accuracy'], 2)} en test, mientras la variante con augmentation obtuvo "
            f"{pct(aug['test_accuracy'], 2)}. La diferencia fue {delta * 100:+.2f} puntos porcentuales.", body),
        table([
            ["Variante", "Val. accuracy", "Test accuracy", "Macro F1", "Diagnostico"],
            ["Sin augmentation", pct(validation_by_model["cnn_regularized_no_aug"]["val_accuracy_checkpoint"]),
             pct(no_aug["test_accuracy"]), pct(no_aug["macro_f1"]),
             validation_by_model["cnn_regularized_no_aug"]["curve_diagnosis"]],
            ["Con augmentation", pct(validation_by_model["cnn_regularized_aug"]["val_accuracy_checkpoint"]),
             pct(aug["test_accuracy"]), pct(aug["macro_f1"]),
             validation_by_model["cnn_regularized_aug"]["curve_diagnosis"]],
        ], [4.0 * cm, 2.6 * cm, 2.6 * cm, 2.3 * cm, 3.95 * cm], font_size=8.0),
        Spacer(1, 0.3 * cm),
        para(
            "Como ambas variantes permanecieron en underfitting, el resultado no implica que el "
            "augmentation sea perjudicial en general. En este presupuesto de entrenamiento, las "
            "transformaciones aumentaron la dificultad de optimizacion y la red propia no aprendio "
            "representaciones suficientemente discriminativas. Una nueva iteracion deberia ajustarse "
            "solo con train y validation, sin reabrir test.", body),
        para("10 Conclusiones", h1),
        para(
            "El objetivo academico se cumplio. MobileNetV2 con fine tuning parcial fue seleccionada "
            "por validation y obtuvo 86.14% de accuracy, 86.43% de precision macro, 85.81% de recall "
            "macro y 85.10% de macro F1 en test. La brecha train-validation del mejor checkpoint fue "
            "2.99 puntos, por debajo del criterio de alerta definido.", body),
        para(
            "La comparacion demuestra que las representaciones preentrenadas fueron esenciales para "
            "este conjunto y presupuesto. El Fully Connected obtuvo 3.53%; la mejor CNN entrenada desde "
            "cero, 25.00%; MobileNetV2 congelada, 84.24%; y el fine tuning agrego 1.90 puntos hasta superar "
            "el umbral. El analisis por clase identifica oportunidades en razas visualmente cercanas, "
            "sin utilizar esa evidencia para modificar el modelo despues de abrir test.", body),
        para("Limitaciones", h2),
        para(
            "El test contiene 368 imagenes, con pocos ejemplos por raza. Una sola particion no cuantifica "
            "la variabilidad entre semillas. Ademas, los umbrales automaticos de overfitting y underfitting "
            "son criterios operativos, no pruebas estadisticas. Los resultados son reproducibles mediante "
            "los checkpoints, historiales y configuracion guardados.", body),
        para("Referencias", h2),
        para(
            "1. Parkhi, O. M. et al. <i>Cats and Dogs</i>. IEEE CVPR, 2012. "
            "<link href='https://www.robots.ox.ac.uk/~vgg/data/pets/' color='#1F6F8B'>Oxford-IIIT Pet Dataset</link>.<br/>"
            "2. TensorFlow Datasets. <link href='https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet' color='#1F6F8B'>Oxford IIIT Pet catalog</link>.<br/>"
            "3. TensorFlow. <link href='https://www.tensorflow.org/tutorials/images/transfer_learning' color='#1F6F8B'>Transfer learning and fine tuning</link>.<br/>"
            "4. Sandler, M. et al. <i>MobileNetV2: Inverted Residuals and Linear Bottlenecks</i>. CVPR, 2018.",
            body,
        ),
    ]

    doc = ReportDoc(OUTPUT)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
