from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from vqa_service import analyze_image, answer_question


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUESTION = "What is shown in this image?"


class VqaSafetyTests(unittest.TestCase):
    def test_actual_image_and_question_are_forwarded_to_model(self):
        image = Image.new("RGB", (48, 32), color=(120, 75, 45))
        question = "What visible skin lumps are present?"
        with patch("vqa_service._run_vqa_model", return_value="visible skin lumps") as run_model:
            result = analyze_image(image, question)

        passed_image, passed_question = run_model.call_args.args
        self.assertEqual(passed_image.size, image.size)
        self.assertEqual(passed_image.tobytes(), image.tobytes())
        self.assertEqual(passed_question, question)
        self.assertEqual(result["image_observations"], "visible skin lumps")
        self.assertEqual(result["status"], "ok")

    def test_shot_claim_is_discarded(self):
        with patch("vqa_service._run_vqa_model", return_value="he got shot"):
            result = analyze_image(Image.new("RGB", (4, 4)), "What disease could these skin lumps indicate?")

        self.assertIsNone(result["image_observations"])
        self.assertEqual(result["status"], "unusable")

    def test_unrelated_generic_and_diagnostic_outputs_are_discarded(self):
        image = Image.new("RGB", (4, 4))
        for model_answer in ("a cow", "mastitis", "give antibiotics"):
            with self.subTest(model_answer=model_answer), patch(
                "vqa_service._run_vqa_model", return_value=model_answer
            ):
                result = analyze_image(image, "What visible skin lumps are present?")
                self.assertIsNone(result["image_observations"])

    def test_udder_features_are_relevant_without_claiming_mastitis(self):
        with patch("vqa_service._run_vqa_model", return_value="swollen udder"):
            result = analyze_image(Image.new("RGB", (4, 4)), "What signs of mastitis can I identify?")

        self.assertEqual(result["image_observations"], "swollen udder")

    def test_legacy_answer_question_shape_and_unusable_response(self):
        image = Image.new("RGB", (4, 4))
        with patch("vqa_service._run_vqa_model", return_value="black and white"):
            result = answer_question(image, "What color is the cow?")
        self.assertEqual(set(result), {"question", "answer"})
        self.assertEqual(result["answer"], "black and white")

        with patch("vqa_service._run_vqa_model", return_value="he got shot"):
            with self.assertRaisesRegex(RuntimeError, "no reliable"):
                answer_question(image, "What is visible?")


def main() -> None:
    image_dir = Path(__file__).resolve().parent / "test_images"
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported sample image found in {image_dir}")

    image_path = image_paths[0]
    with Image.open(image_path) as source_image:
        result = answer_question(source_image, QUESTION)

    print(f"Image: {image_path}")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
