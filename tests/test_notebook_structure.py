import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Actividad_grupal_SCA.ipynb"


class NotebookQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.data["cells"]
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "markdown"
        )
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.cells
            if cell.get("cell_type") == "code"
        )

    def test_notebook_format(self):
        self.assertEqual(self.data["nbformat"], 4)
        self.assertGreaterEqual(len(self.cells), 30)

    def test_every_code_cell_has_valid_python_syntax(self):
        for index, cell in enumerate(self.cells):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                self.fail(f"Invalid Python syntax in cell {index}: {exc}")

    def test_required_sections_exist(self):
        required = [
            "Análisis exploratorio",
            "Diseño experimental",
            "Fully Connected",
            "CNN propia",
            "Data augmentation",
            "Transfer learning",
            "Fine tuning",
            "Evaluación final",
            "Análisis de errores",
            "Conclusiones",
        ]
        for heading in required:
            self.assertIn(heading, self.markdown)

    def test_all_experiments_are_declared(self):
        experiment_ids = [
            "fc_baseline",
            "cnn_baseline",
            "cnn_regularized_no_aug",
            "cnn_regularized_aug",
            "mobilenet_transfer",
            "mobilenet_finetuned",
        ]
        for experiment_id in experiment_ids:
            self.assertIn(experiment_id, self.code)

    def test_split_uses_percentages_not_eighty_examples(self):
        self.assertIn("train[:80%]", self.code)
        self.assertNotIn("'train[:80]'", self.code)
        self.assertNotIn('"train[:80]"', self.code)

    def test_test_evaluation_happens_after_selection(self):
        selection = self.code.index("best_validation_model")
        final_test = self.code.index("FINAL TEST GATE")
        self.assertLess(selection, final_test)


if __name__ == "__main__":
    unittest.main()
