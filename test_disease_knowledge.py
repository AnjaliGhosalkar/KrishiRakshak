import hashlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from disease_knowledge import DiseaseKnowledgeBase, KnowledgeServiceError


class TokenEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:2], "little") % vectors.shape[1]] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class DiseaseKnowledgeTests(unittest.TestCase):
    def test_workspace_pdfs_extract_to_page_tagged_chunks(self):
        project_dir = Path(__file__).resolve().parent
        knowledge = DiseaseKnowledgeBase()
        chunks, report = knowledge._extract_documents()

        self.assertEqual(report["scanned"], 7)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(report["indexed"])
        self.assertTrue(all(chunk["page"] >= 1 for chunk in chunks))
        self.assertTrue(all((project_dir / "disease_documents" / chunk["source"]).is_file() for chunk in chunks))

    def test_empty_corrupt_and_unsupported_files_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory) / "docs"
            documents_dir.mkdir()
            (documents_dir / "empty.pdf").write_bytes(b"")
            (documents_dir / "corrupt.pdf").write_bytes(b"not a PDF")
            (documents_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            knowledge = DiseaseKnowledgeBase(documents_dir=documents_dir)

            chunks, report = knowledge._extract_documents()

            self.assertEqual(chunks, [])
            self.assertEqual(report["scanned"], 2)
            self.assertEqual({item["filename"] for item in report["skipped"]}, {"empty.pdf", "corrupt.pdf"})

    def test_unreadable_pdf_is_reported_without_aborting_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            documents_dir = Path(directory) / "docs"
            documents_dir.mkdir()
            (documents_dir / "locked.pdf").write_bytes(b"placeholder")
            knowledge = DiseaseKnowledgeBase(documents_dir=documents_dir)
            with patch("pypdf.PdfReader", side_effect=PermissionError("access denied")):
                chunks, report = knowledge._extract_documents()

            self.assertEqual(chunks, [])
            self.assertEqual(report["skipped"][0]["filename"], "locked.pdf")
            self.assertIn("access denied", report["skipped"][0]["reason"])

    def test_retrieval_cites_passage_and_survives_index_reload(self):
        chunks = [
            {"text": "Mastitis may cause inflammation of the udder.", "source": "Mastitis.pdf", "page": 3},
            {"text": "Foot disorders can cause lameness.", "source": "cattle-diseases-farmers-guide.pdf", "page": 8},
        ]
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                documents_dir=Path(directory) / "docs",
                index_dir=Path(directory) / "index",
                embedder=TokenEmbedder(),
            )
            with patch.object(knowledge, "_extract_documents", return_value=(chunks, {"scanned": 2})):
                report = knowledge.reindex()
            self.assertTrue(report["built"])

            reloaded = DiseaseKnowledgeBase(
                documents_dir=Path(directory) / "docs",
                index_dir=Path(directory) / "index",
                embedder=TokenEmbedder(),
            )
            answer = reloaded.ask("mastitis inflammation", min_score=0.3)

            self.assertEqual(answer["answer_mode"], "llm_not_configured")
            self.assertIn("AI answer generation is not configured yet", answer["answer"])
            self.assertNotIn("Mastitis may cause inflammation", answer["answer"])
            self.assertEqual(answer["sources"][0]["filename"], "Mastitis.pdf")
            self.assertEqual(answer["sources"][0]["page"], 3)
            self.assertIn("Mastitis may cause inflammation", answer["sources"][0]["excerpt"])
            self.assertIn(chunks[0]["text"], answer["passages"][0]["text"])

    def test_configured_generator_uses_evidence_and_follow_up_history(self):
        chunks = [{"text": "Mastitis may cause inflammation of the udder.", "source": "Mastitis.pdf", "page": 3}]
        generator_calls = []

        def answer_generator(system_prompt, user_prompt):
            generator_calls.append((system_prompt, user_prompt))
            return "Mastitis inflames the udder [1]."

        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
                answer_generator=answer_generator,
            )
            with patch.object(knowledge, "_extract_documents", return_value=(chunks, {})):
                knowledge.reindex()

            answer = knowledge.ask(
                "and prevention?",
                min_score=0.0,
                history=[{"role": "user", "content": "Tell me about mastitis."}],
            )

            self.assertEqual(answer["answer_mode"], "generated")
            self.assertEqual(answer["answer"], "Mastitis inflames the udder [1].")
            self.assertIn("Mastitis.pdf, page 3", generator_calls[0][1])
            self.assertIn("Tell me about mastitis.", generator_calls[0][1])
            self.assertIn("Never invent", generator_calls[0][0])

    def test_uncited_or_unrelated_generation_is_rejected_not_replaced_by_pdf_text(self):
        chunk = {"text": "Mastitis may cause inflammation of the udder.", "source": "Mastitis.pdf", "page": 3}
        for generated in ["Mastitis affects the udder.", "Mastitis affects the udder [9].", "he got shot [1]."]:
            with self.subTest(generated=generated), tempfile.TemporaryDirectory() as directory:
                knowledge = DiseaseKnowledgeBase(
                    index_dir=directory,
                    embedder=TokenEmbedder(),
                    answer_generator=lambda system, user: generated,
                )
                with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                    knowledge.reindex()

                result = knowledge.ask("mastitis inflammation", min_score=0.3)

                self.assertEqual(result["answer_mode"], "generation_rejected")
                self.assertNotIn("Mastitis may cause inflammation", result["answer"])
                self.assertIn("Mastitis.pdf", result["sources"][0]["filename"])
                self.assertNotIn("he got shot", result["answer"])

    def test_configured_openai_compatible_endpoint_receives_evidence(self):
        chunk = {"text": "Mastitis may cause inflammation of the udder.", "source": "Mastitis.pdf", "page": 3}
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory, embedder=TokenEmbedder())
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            with patch.dict(os.environ, {
                "OPENAI_BASE_URL": "http://local-model.test/v1",
                "OPENAI_MODEL": "local-vet-model",
                "OPENAI_API_KEY": "test-placeholder",
            }), patch("requests.post") as post:
                post.return_value.json.return_value = {
                    "choices": [{"message": {"content": "Mastitis can inflame the udder [1]."}}]
                }
                result = knowledge.ask("mastitis inflammation", min_score=0.3)

            self.assertEqual(result["answer_mode"], "generated")
            self.assertEqual(result["answer"], "Mastitis can inflame the udder [1].")
            self.assertEqual(post.call_args.args[0], "http://local-model.test/v1/chat/completions")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-placeholder")
            self.assertIn("Mastitis.pdf, page 3", post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_missing_llm_configuration_never_uses_retrieved_text_as_answer(self):
        chunk = {"text": "Mastitis can cause inflammation of the udder.", "source": "Mastitis.pdf", "page": 3}
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory, embedder=TokenEmbedder())
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": "", "OPENAI_BASE_URL": ""}):
                result = knowledge.ask("mastitis inflammation", min_score=0.0)

        self.assertEqual(result["answer_mode"], "llm_not_configured")
        self.assertEqual(
            result["answer"],
            "AI answer generation is not configured yet. Please configure the language model to generate answers from the veterinary knowledge base.",
        )
        self.assertNotIn("he got shot", result["answer"])

    def test_unmatched_question_returns_no_sources_or_claims(self):
        chunks = [{"text": "Mastitis may cause udder inflammation.", "source": "Mastitis.pdf", "page": 2}]
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
            )
            with patch.object(knowledge, "_extract_documents", return_value=(chunks, {})):
                knowledge.reindex()

            result = knowledge.ask("What is the capital of France?", min_score=0.5)

            self.assertEqual(result["passages"], [])
            self.assertEqual(result["sources"], [])
            self.assertEqual(result["answer_mode"], "out_of_scope")
            self.assertIn("outside my knowledge scope", result["answer"])

    def test_standalone_off_topic_question_does_not_inherit_old_disease_context(self):
        knowledge = DiseaseKnowledgeBase()
        result = knowledge.ask(
            "What is the capital of France?",
            history=[{"role": "user", "content": "What are the signs of mastitis?"}],
        )

        self.assertEqual(result["answer_mode"], "out_of_scope")
        self.assertEqual(result["sources"], [])

    def test_unsupported_veterinary_question_returns_not_enough_evidence(self):
        chunks = [{"text": "Mastitis may cause udder inflammation.", "source": "Mastitis.pdf", "page": 2}]
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory, embedder=TokenEmbedder())
            with patch.object(knowledge, "_extract_documents", return_value=(chunks, {})):
                knowledge.reindex()

            result = knowledge.ask("What is the exact dosage of medicine X?")

        self.assertEqual(result["passages"], [])
        self.assertEqual(result["answer_mode"], "no_evidence")
        self.assertIn("couldn't find enough information", result["answer"])

    def test_empty_question_and_missing_index_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory, embedder=TokenEmbedder())
            with self.assertRaises(ValueError):
                knowledge.ask("   ")
            with self.assertRaises(KnowledgeServiceError):
                knowledge.ask("mastitis")

    def test_embedding_model_failure_is_reported(self):
        chunk = {"text": "Mastitis information.", "source": "Mastitis.pdf", "page": 1}
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory)
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                with patch.object(
                    knowledge,
                    "_get_embedder",
                    side_effect=KnowledgeServiceError("model unavailable"),
                ):
                    with self.assertRaisesRegex(KnowledgeServiceError, "model unavailable"):
                        knowledge.reindex()

    def test_lumpy_skin_mastitis_and_fmd_questions_retrieve_matching_sources(self):
        cases = [
            (
                "What signs are associated with lumpy skin disease?",
                "Lumpy skin disease may cause firm nodules on the cattle skin.",
                "Lumpy-Skin-Disease.pdf",
                "Lumpy skin disease may cause firm skin nodules [1].",
            ),
            (
                "What signs can mastitis cause in cattle?",
                "Mastitis causes inflammation and swelling of the udder.",
                "Mastitis.pdf",
                "Mastitis can cause udder inflammation [1].",
            ),
            (
                "What are signs of foot-and-mouth disease?",
                "Foot and mouth disease can cause lesions in the mouth and on feet.",
                "foot-and-mouth-disease.pdf",
                "Foot-and-mouth disease can cause mouth lesions [1].",
            ),
        ]
        for question, text, filename, generated in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                knowledge = DiseaseKnowledgeBase(
                    index_dir=directory,
                    embedder=TokenEmbedder(),
                    answer_generator=lambda system, user, generated=generated: generated,
                )
                chunks = [{"text": text, "source": filename, "page": 2}]
                with patch.object(knowledge, "_extract_documents", return_value=(chunks, {})):
                    knowledge.reindex()

                result = knowledge.ask(question, min_score=0.0)

                self.assertEqual(result["answer_mode"], "generated")
                self.assertEqual(result["sources"][0]["filename"], filename)
                self.assertIn("[1]", result["answer"])

    def test_image_observations_documents_and_prediction_reach_generator(self):
        chunk = {
            "text": "Lumpy skin disease may cause firm nodules on cattle skin.",
            "source": "Lumpy-Skin-Disease.pdf",
            "page": 17,
        }
        generator_calls = []

        def answer_generator(system_prompt, user_prompt):
            generator_calls.append((system_prompt, user_prompt))
            return (
                "Visual observation: the image model suggests visible skin nodules.\n"
                "The reference describes nodules associated with lumpy skin disease [1]."
            )

        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
                answer_generator=answer_generator,
            )
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            result = knowledge.ask(
                "What disease could these skin lumps indicate?",
                min_score=0.0,
                image_observations="visible skin nodules",
                prediction_context={"disease": "Lumpy", "confidence": "61.00%"},
                image_provided=True,
            )

        self.assertEqual(result["answer_mode"], "generated")
        self.assertIn("visible skin nodules", generator_calls[0][1])
        self.assertIn("Lumpy-Skin-Disease.pdf, page 17", generator_calls[0][1])
        self.assertIn('"confidence": "61.00%"', generator_calls[0][1])
        self.assertIn("not a diagnosis", generator_calls[0][0])
        self.assertEqual(result["image_observations"], "visible skin nodules")

    def test_image_udder_question_uses_mastitis_evidence_cautiously(self):
        chunk = {
            "text": "Mastitis may cause inflammation and swelling of the udder.",
            "source": "Mastitis.pdf",
            "page": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
                answer_generator=lambda system, user: (
                    "Visual observation: the image model suggests udder swelling.\n"
                    "Mastitis references describe udder inflammation [1]."
                ),
            )
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            result = knowledge.ask(
                "What visible udder abnormalities can I identify?",
                min_score=0.0,
                image_observations="possible udder swelling",
                image_provided=True,
            )

        self.assertEqual(result["answer_mode"], "generated")
        self.assertEqual(result["sources"][0]["filename"], "Mastitis.pdf")
        self.assertIn("udder swelling", result["answer"])

    def test_image_observation_can_ground_an_underspecified_question(self):
        chunk = {
            "text": "Lumpy skin disease may cause firm nodules on cattle skin.",
            "source": "Lumpy-Skin-Disease.pdf",
            "page": 17,
        }
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
                answer_generator=lambda system, user: "Skin nodules are described [1].",
            )
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            result = knowledge.ask(
                "What is this?",
                min_score=0.0,
                image_observations="visible skin nodules",
                image_provided=True,
            )

        self.assertEqual(result["answer_mode"], "generated")
        self.assertEqual(result["sources"][0]["filename"], "Lumpy-Skin-Disease.pdf")

    def test_generated_visual_details_not_in_observation_are_rejected(self):
        chunk = {
            "text": "Lumpy skin disease may cause firm nodules on cattle skin.",
            "source": "Lumpy-Skin-Disease.pdf",
            "page": 17,
        }
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(
                index_dir=directory,
                embedder=TokenEmbedder(),
                answer_generator=lambda system, user: "Visual observation: red and bleeding skin nodules are visible.",
            )
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            result = knowledge.ask(
                "What are these skin lumps?",
                min_score=0.0,
                image_observations="visible skin nodules",
                image_provided=True,
            )

        self.assertEqual(result["answer_mode"], "generation_rejected")
        self.assertNotIn("red", result["answer"])

    def test_inconclusive_image_and_unrelated_question_do_not_force_disease(self):
        knowledge = DiseaseKnowledgeBase()
        result = knowledge.ask(
            "What is in the background?",
            image_provided=True,
        )

        self.assertEqual(result["answer_mode"], "out_of_scope")
        self.assertEqual(result["sources"], [])
        self.assertNotIn("diagnosis", result["answer"].casefold())

    def test_filename_alone_cannot_make_irrelevant_passage_relevant(self):
        chunk = {"text": "A farmer repaired a fence beside a road.", "source": "Mastitis.pdf", "page": 12}
        with tempfile.TemporaryDirectory() as directory:
            knowledge = DiseaseKnowledgeBase(index_dir=directory, embedder=TokenEmbedder())
            with patch.object(knowledge, "_extract_documents", return_value=([chunk], {})):
                knowledge.reindex()

            result = knowledge.ask("What are signs of mastitis?", min_score=0.0)

        self.assertEqual(result["answer_mode"], "no_evidence")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()