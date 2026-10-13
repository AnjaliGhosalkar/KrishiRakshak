import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import app as flask_app
from disease_knowledge import DiseaseKnowledgeBase, KnowledgeServiceError


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=(120, 75, 45)).save(buffer, format="PNG")
    return buffer.getvalue()


class AssistantApiTests(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.app.test_client()

    def test_text_question_uses_existing_knowledge_endpoint_contract(self):
        knowledge = Mock()
        knowledge.ask.return_value = {
            "question": "What are signs of mastitis?",
            "answer": "The documents describe udder inflammation [1].",
            "answer_mode": "generated",
            "status": "generated",
            "sources": [{"filename": "Mastitis.pdf", "page": 2}],
        }
        with patch.object(flask_app, "knowledge_base", knowledge):
            response = self.client.post(
                "/knowledge/ask",
                json={"question": "What are signs of mastitis?", "history": []},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer_mode"], "generated")
        self.assertEqual(response.json["sources"][0]["filename"], "Mastitis.pdf")
        knowledge.ask.assert_called_once_with(
            "What are signs of mastitis?",
            top_k=4,
            history=[],
            image_observations=None,
            prediction_context=None,
            image_provided=False,
        )

    def test_image_question_forwards_decoded_image_question_and_prediction(self):
        knowledge = Mock()
        knowledge.ask.return_value = {
            "question": "What could these skin lumps indicate?",
            "answer": "A photograph cannot confirm a disease [1].",
            "answer_mode": "generated",
            "status": "generated",
            "sources": [],
        }
        prediction_context = {"disease": "Lumpy", "confidence": "61.00%"}
        with patch.object(flask_app, "knowledge_base", knowledge), patch.object(
            flask_app,
            "analyze_image",
            return_value={"status": "ok", "image_observations": "visible skin nodules"},
        ) as analyze:
            response = self.client.post(
                "/knowledge/ask",
                data={
                    "question": "What could these skin lumps indicate?",
                    "history": "[]",
                    "prediction_context": json.dumps(prediction_context),
                    "image": (BytesIO(png_bytes()), "cattle.png"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["image_observations"], "visible skin nodules")
        self.assertEqual(response.json["image_analysis_status"], "ok")
        image_arg, question_arg = analyze.call_args.args
        self.assertEqual(question_arg, "What could these skin lumps indicate?")
        self.assertEqual(image_arg.size, (16, 12))
        knowledge.ask.assert_called_once_with(
            "What could these skin lumps indicate?",
            top_k=4,
            history=[],
            image_observations="visible skin nodules",
            prediction_context=prediction_context,
            image_provided=True,
        )

    def test_unusable_image_observation_does_not_become_answer(self):
        knowledge = Mock()
        knowledge.ask.return_value = {
            "question": "What is in the background?",
            "answer": "That question is outside my knowledge scope.",
            "answer_mode": "out_of_scope",
            "status": "out_of_scope",
            "sources": [],
        }
        with patch.object(flask_app, "knowledge_base", knowledge), patch.object(
            flask_app,
            "analyze_image",
            return_value={"status": "unusable", "image_observations": None},
        ):
            response = self.client.post(
                "/knowledge/ask",
                data={
                    "question": "What is in the background?",
                    "image": (BytesIO(png_bytes()), "cattle.png"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["image_analysis_status"], "unusable")
        self.assertNotIn("image_observations", response.json)
        self.assertEqual(response.json["answer_mode"], "out_of_scope")
        self.assertIsNone(knowledge.ask.call_args.kwargs["image_observations"])

    def test_knowledge_service_failure_is_not_reported_as_image_failure(self):
        knowledge = Mock()
        knowledge.ask.side_effect = KnowledgeServiceError("configured answer provider unavailable")
        with patch.object(flask_app, "knowledge_base", knowledge):
            response = self.client.post("/knowledge/ask", json={"question": "What are signs of mastitis?"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"], "configured answer provider unavailable")

    def test_empty_question_corrupt_image_and_unsupported_file_are_rejected(self):
        empty_question = self.client.post(
            "/knowledge/ask",
            data={"question": "  "},
            content_type="multipart/form-data",
        )
        corrupt_image = self.client.post(
            "/knowledge/ask",
            data={"question": "What is visible?", "image": (BytesIO(b"not an image"), "cow.png")},
            content_type="multipart/form-data",
        )
        unsupported_image = self.client.post(
            "/knowledge/ask",
            data={"question": "What is visible?", "image": (BytesIO(png_bytes()), "cow.gif")},
            content_type="multipart/form-data",
        )

        self.assertEqual(empty_question.status_code, 400)
        self.assertIn("question", empty_question.json["error"].lower())
        self.assertEqual(corrupt_image.status_code, 400)
        self.assertIn("decodable", corrupt_image.json["error"])
        self.assertEqual(unsupported_image.status_code, 400)
        self.assertIn("JPEG, PNG, WEBP, or BMP", unsupported_image.json["error"])

    def test_legacy_vqa_route_preserves_question_answer_shape(self):
        with patch.object(
            flask_app,
            "answer_question",
            return_value={"question": "What is visible?", "answer": "brown cattle"},
        ):
            response = self.client.post(
                "/vqa",
                data={
                    "question": "What is visible?",
                    "image": (BytesIO(png_bytes()), "cow.png"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json), {"question", "answer"})
        self.assertEqual(response.json["answer"], "brown cattle")

    def test_predict_route_keeps_legacy_response_fields(self):
        extractor = Mock()
        extractor.predict.return_value = np.zeros((1, 1, 1, 1), dtype=np.float32)
        scaler = Mock()
        scaler.transform.return_value = np.zeros((1, 1), dtype=np.float32)
        classifier = Mock()
        classifier.predict_proba.return_value = np.asarray([[0.99]])
        classifier.predict.return_value = np.asarray([0])
        encoder = Mock()
        encoder.inverse_transform.return_value = ["healthy"]
        ebdre = {
            "cnn_confidence": 99.0,
            "clinical_support_score": None,
            "evidence_strength": "limited",
            "available_evidence": [],
            "confidence_level": "high",
            "ranked_evidence": [],
            "clinical_summary": None,
            "biomarkers": None,
            "evidence_matching": None,
        }
        with (
            patch.object(flask_app, "feature_extractor", extractor),
            patch.object(flask_app, "scaler", scaler),
            patch.object(flask_app, "rf", classifier),
            patch.object(flask_app, "le", encoder),
            patch.object(flask_app, "get_progression_images", return_value={}),
            patch.object(flask_app, "safe_run_ebdre_pipeline", return_value=ebdre),
            patch.object(flask_app, "extract_biomarkers", return_value={}),
            patch.object(flask_app, "describe_image", return_value="a cow in the image"),
            patch.object(flask_app, "explain_rf_prediction", side_effect=RuntimeError("mocked")),
            patch.object(flask_app, "explain_saved_mobilenet", side_effect=RuntimeError("mocked")),
        ):
            response = self.client.post(
                "/predict",
                data={"image": (BytesIO(png_bytes()), "cow.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["disease"], "Healthy")
        self.assertEqual(response.json["stage"], "None")
        self.assertEqual(response.json["confidence"], "99.00%")
        self.assertIn("recommendation", response.json)
        self.assertIn("xai", response.json)
        self.assertEqual(response.json["cnn_confidence"], 99.0)


if __name__ == "__main__":
    unittest.main()
