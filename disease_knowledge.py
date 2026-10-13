import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

os.environ.setdefault("USE_TF", "0")

import numpy as np


class KnowledgeServiceError(RuntimeError):
    """Raised when the document index or embedding service is unavailable."""


class DiseaseKnowledgeBase:
    def __init__(
        self,
        documents_dir=None,
        index_dir=None,
        model_name=None,
        embedder=None,
        answer_generator=None,
    ):
        project_dir = Path(__file__).resolve().parent
        self.documents_dir = Path(documents_dir or project_dir / "disease_documents")
        self.index_dir = Path(index_dir or project_dir / "disease_knowledge")
        self.model_name = model_name or os.environ.get(
            "KRISHIRAKSHAK_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._embedder = embedder
        self._answer_generator = answer_generator
        self._chunks = []
        self._vectors = None
        self._index_loaded = False
        self._load_error = None
        self._lock = RLock()

    @property
    def _vectors_path(self):
        return self.index_dir / "vectors.npz"

    @property
    def _metadata_path(self):
        return self.index_dir / "metadata.json"

    @staticmethod
    def _split_page(text, chunk_size=1100, overlap=180):
        normalized = " ".join((text or "").split())
        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            content = normalized[start:end].strip()
            if content:
                chunks.append(content)
            if end >= len(normalized):
                break
            next_start = max(start + 1, end - overlap)
            while next_start < len(normalized) and normalized[next_start] != " ":
                next_start += 1
            start = min(next_start + 1, len(normalized))
        return chunks

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise KnowledgeServiceError(
                    "The embedding model is unavailable. Install the knowledge requirements "
                    "and allow the model to download on first use."
                ) from exc
        return self._embedder

    def _encode(self, texts):
        try:
            encoded = self._get_embedder().encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            vectors = np.asarray(encoded, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts):
                raise ValueError("Embedding model returned an unexpected vector shape")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.all(np.isfinite(vectors)):
                raise ValueError("Embedding model returned invalid vectors")
            return vectors / norms
        except KnowledgeServiceError:
            raise
        except Exception as exc:
            raise KnowledgeServiceError("Could not create semantic embeddings") from exc

    def _extract_documents(self):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise KnowledgeServiceError(
                "PDF extraction is unavailable. Install the knowledge requirements."
            ) from exc

        report = {"scanned": 0, "indexed": [], "skipped": []}
        chunks = []
        if not self.documents_dir.is_dir():
            return chunks, report

        pdf_paths = sorted(
            path for path in self.documents_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        report["scanned"] = len(pdf_paths)
        for path in pdf_paths:
            try:
                if path.stat().st_size == 0:
                    raise ValueError("empty file")
                reader = PdfReader(str(path), strict=False)
                if reader.is_encrypted and not reader.decrypt(""):
                    raise ValueError("password-protected PDF")
                document_chunks = []
                for page_number, page in enumerate(reader.pages, start=1):
                    for text in self._split_page(page.extract_text() or ""):
                        document_chunks.append({
                            "text": text,
                            "source": path.name,
                            "page": page_number,
                        })
                if not document_chunks:
                    raise ValueError("no extractable text")
                chunks.extend(document_chunks)
                report["indexed"].append({"filename": path.name, "chunks": len(document_chunks)})
            except Exception as exc:
                report["skipped"].append({"filename": path.name, "reason": str(exc)})
        return chunks, report

    def _write_index(self, vectors, chunks, report):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": 1,
            "model_name": self.model_name,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks,
            "report": report,
        }
        vector_temp = None
        metadata_temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.index_dir, suffix=".npz", delete=False) as handle:
                vector_temp = Path(handle.name)
                np.savez_compressed(handle, vectors=vectors)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.index_dir, suffix=".json", delete=False
            ) as handle:
                metadata_temp = Path(handle.name)
                json.dump(metadata, handle, ensure_ascii=False)
            os.replace(vector_temp, self._vectors_path)
            os.replace(metadata_temp, self._metadata_path)
        finally:
            for temporary_path in (vector_temp, metadata_temp):
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()

    def _load_index(self):
        if self._index_loaded:
            return
        with self._lock:
            if self._index_loaded:
                return
            if not self._vectors_path.is_file() or not self._metadata_path.is_file():
                return
            try:
                metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                with np.load(self._vectors_path, allow_pickle=False) as archive:
                    vectors = np.asarray(archive["vectors"], dtype=np.float32)
                chunks = metadata["chunks"]
                if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                    raise ValueError("Index metadata and vector counts do not match")
                self._chunks = chunks
                self._vectors = vectors
                self._index_loaded = True
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise KnowledgeServiceError("The persisted disease knowledge index is invalid") from exc

    def reindex(self):
        with self._lock:
            chunks, report = self._extract_documents()
            if not chunks:
                report["built"] = False
                report["error"] = "No readable PDF text was found in disease_documents."
                return report
            vectors = self._encode([chunk["text"] for chunk in chunks])
            self._write_index(vectors, chunks, report)
            self._chunks = chunks
            self._vectors = vectors
            self._index_loaded = True
            self._load_error = None
            report["built"] = True
            report["chunk_count"] = len(chunks)
            return report

    def status(self):
        pdf_files = sorted(
            path.name for path in self.documents_dir.glob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ) if self.documents_dir.is_dir() else []
        try:
            self._load_index()
        except KnowledgeServiceError:
            pass
        return {
            "documents_dir": str(self.documents_dir),
            "documents": pdf_files,
            "indexed": self._index_loaded,
            "chunk_count": len(self._chunks),
            "model": self.model_name,
            "index_error": self._load_error,
        }

    @staticmethod
    def _history_for_generation(history):
        if history is None:
            return []
        if not isinstance(history, list):
            raise ValueError("history must be a list of user and assistant messages")
        validated = []
        for message in history[-8:]:
            if not isinstance(message, dict):
                raise ValueError("history messages must include a role and content")
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("history messages must include a valid role and text content")
            validated.append({"role": role, "content": content[:2000]})
        return validated

    @staticmethod
    def _build_sources(passages):
        sources = []
        source_ids = {}
        for passage in passages:
            key = (passage["source"], passage["page"])
            if key not in source_ids:
                source_id = len(sources) + 1
                source_ids[key] = source_id
                sources.append({
                    "id": source_id,
                    "filename": passage["source"],
                    "page": passage["page"],
                    "excerpt": passage["text"],
                    "score": passage["score"],
                })
        return sources

    @staticmethod
    def _is_veterinary_question(question):
        return bool(re.search(
            r"\b(cattle|cow|cows|calf|calves|bovine|livestock|herd|udder|hoof|hooves|"
            r"mastitis|disease|diseases|symptom|symptoms|signs?|treatment|treat|medicine|"
            r"medication|drug|dosage|dose|prevention|prevent|veterinar\w*|lumpy|fmd|"
            r"infection|infectious|milk production|foot and mouth|mouth disease|skin|nodules?|lumps?|lesions?)\b",
            question,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _is_follow_up(question):
        normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
        return len(normalized.split()) <= 6 and bool(re.match(
            r"^(and\b|what about\b|how about\b|why\b|when\b|does it\b|can it\b|is it\b|what if\b|it\b|they\b|those\b)",
            normalized,
        ))

    @staticmethod
    def _topic_for_question(question):
        normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
        words = set(normalized.split())
        if "mastitis" in words:
            return "mastitis"
        if (
            "lsd" in words
            or {"lumpy", "skin"}.issubset(words)
            or bool(words & {"nodule", "nodules", "lump", "lumps"} and "skin" in words)
        ):
            return "lumpy_skin"
        if "fmd" in words or {"foot", "mouth"}.issubset(words):
            return "foot_mouth"
        if words & {"udder", "teat", "teats"}:
            return "mastitis"
        if words & {"hoof", "hooves", "lameness"} or {"foot", "disorders"}.issubset(words):
            return "hoof"
        return None

    @staticmethod
    def _matches_topic(chunk, topic):
        content = chunk["text"].casefold()
        if topic == "mastitis":
            return bool(re.search(r"\b(mastitis|udder|teat|teats|milk gland|mammary)\b", content))
        if topic == "lumpy_skin":
            return bool(re.search(
                r"\b(lumpy skin|lsd|skin nodules?|skin lumps?|skin lesions?|cutaneous nodules?)\b",
                content,
            ))
        if topic == "foot_mouth":
            return bool(re.search(
                r"\b(foot and mouth|fmd|mouth lesions?|vesicles?|blisters?)\b",
                content,
            ))
        if topic == "hoof":
            return bool(re.search(r"\b(hoof|hooves|lameness|foot abscess|foot rot)\b", content))
        return True

    @staticmethod
    def _generation_prompts(
        question,
        sources,
        history,
        image_observations=None,
        prediction_context=None,
        image_provided=False,
    ):
        system_prompt = (
            "You are KrishiRakshak AI, a cattle-health knowledge assistant. Directly answer the user's actual "
            "question in clear, simple English. Use only relevant veterinary facts in DOCUMENT CONTEXT; do not "
            "use outside veterinary knowledge. "
            "IMAGE OBSERVATIONS are uncertain outputs from a general-purpose visual model, not confirmed "
            "findings; use them only as cautious possibilities and never turn them into a diagnosis. PREDICTION "
            "CONTEXT is an unconfirmed classifier output, not a diagnosis; do not force the answer to agree with "
            "it, especially when confidence or clinical support is low. Separate what may be visible from what "
            "the documents say. Never invent image details, symptoms, diagnoses, medicines, treatments, "
            "procedures, or dosages. Do not rely on a disease name alone or irrelevant passages. If the image "
            "does not support the requested observation, say it cannot be determined from the image. If the "
            "documents do not support an answer, say: 'I couldn't find enough information about this in the "
            "available veterinary references.' If the image, prediction, and documents disagree, acknowledge "
            "the uncertainty. Cite document-based factual claims using only supplied source IDs, such as [1]. "
            "Never imply that a photograph confirms disease. Recommend a qualified veterinarian for diagnosis "
            "or treatment decisions and for serious or suspected contagious disease."
        )
        history_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in history
        ) or "(No earlier conversation.)"
        evidence_text = "\n\n".join(
            f"[{source['id']}] {source['filename']}, page {source['page']}: {source['excerpt']}"
            for source in sources
        ) or "(No relevant document passages were retrieved.)"
        image_text = image_observations or (
            "An image was supplied, but image analysis yielded no reliable observation relevant to the question."
            if image_provided
            else "(No image was supplied.)"
        )
        prediction_text = json.dumps(prediction_context, ensure_ascii=True) if prediction_context else "(No existing prediction was supplied.)"
        user_prompt = (
            f"Recent conversation:\n{history_text}\n\n"
            f"DOCUMENT CONTEXT:\n{evidence_text}\n\n"
            f"IMAGE OBSERVATIONS (uncertain; may be empty):\n{image_text}\n\n"
            f"EXISTING PREDICTION CONTEXT (unconfirmed; may be empty):\n{prediction_text}\n\n"
            f"USER QUESTION:\n{question}\n\nGenerate a concise answer."
        )
        return system_prompt, user_prompt

    @staticmethod
    def _llm_configuration():
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        return api_key, model, base_url

    def _generate_answer(
        self,
        question,
        sources,
        history,
        image_observations=None,
        prediction_context=None,
        image_provided=False,
    ):
        system_prompt, user_prompt = self._generation_prompts(
            question, sources, history, image_observations, prediction_context, image_provided
        )
        if self._answer_generator is not None:
            generated = self._answer_generator(system_prompt, user_prompt)
            if isinstance(generated, str) and generated.strip():
                generated = generated.strip()
                if (
                    self._has_valid_citations(
                        generated, sources, allow_uncited_visual=bool(image_observations or image_provided)
                    )
                    and self._is_grounded_in_sources(
                        generated, sources, allow_uncited_visual=bool(image_observations or image_provided)
                    )
                    and self._visual_claims_are_supported(generated, image_observations)
                ):
                    return generated, "generated", None
                return "I couldn't produce a grounded answer from the available veterinary references. Please try rephrasing your question.", "generation_rejected", None
            raise KnowledgeServiceError("The configured answer model returned an empty response")

        api_key, model, base_url = self._llm_configuration()
        if not api_key or not model:
            return (
                "AI answer generation is not configured yet. Please configure the language model to generate answers from the veterinary knowledge base.",
                "llm_not_configured",
                None,
            )

        try:
            import requests

            endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                endpoint,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                },
                timeout=45,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty answer")
            answer = answer.strip()
            if (
                self._has_valid_citations(
                    answer, sources, allow_uncited_visual=bool(image_observations or image_provided)
                )
                and self._is_grounded_in_sources(
                    answer, sources, allow_uncited_visual=bool(image_observations or image_provided)
                )
                and self._visual_claims_are_supported(answer, image_observations)
            ):
                return answer, "generated", None
            return "I couldn't produce a grounded answer from the available veterinary references. Please try rephrasing your question.", "generation_rejected", None
        except Exception as exc:
            raise KnowledgeServiceError("The configured answer-generation service could not be reached") from exc

    @staticmethod
    def _is_visual_statement(sentence):
        normalized = re.sub(r"^[\s>*-]+", "", sentence.casefold())
        return bool(re.match(
            r"^(visual observation(?:s)?|image observation(?:s)?|visible|in the image|the image|"
            r"from the image|this photo|the photo|photograph|i cannot determine from the image|"
            r"it cannot be determined from the image)\b",
            normalized,
        ))

    @staticmethod
    def _visual_claims_are_supported(answer, image_observations):
        framing_words = {
            "a", "an", "and", "appears", "are", "as", "can", "could", "image", "in", "is",
            "it", "may", "model", "observation", "observations", "of", "on", "possibly", "possible",
            "seen", "shows", "suggests", "the", "there", "this", "visible", "visual", "was", "with",
        }

        def content_terms(text):
            return {
                token for token in re.findall(r"[a-z0-9]+", text.casefold())
                if len(token) > 2 and token not in framing_words
            }

        sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
        for sentence in sentences:
            clean_sentence = re.sub(r"\[\d+\]", "", sentence).strip(" -*#\t")
            if DiseaseKnowledgeBase._is_visual_statement(clean_sentence):
                claim_terms = content_terms(clean_sentence)
                if not image_observations or not claim_terms:
                    return False
                if not claim_terms.issubset(content_terms(image_observations)):
                    return False
        return True

    @staticmethod
    def _has_valid_citations(answer, sources, allow_uncited_visual=False):
        source_ids = {str(source["id"]) for source in sources}
        cited_ids = set(re.findall(r"\[(\d+)\]", answer))
        substantive_lines = [
            line.strip()
            for line in answer.splitlines()
            if line.strip() and not re.match(r"^#{1,3}\s+", line.strip())
        ]
        required_citations = [
            line for line in substantive_lines
            if not (allow_uncited_visual and DiseaseKnowledgeBase._is_visual_statement(line))
        ]
        return cited_ids.issubset(source_ids) and all(
            re.search(r"\[\d+\]", line) for line in required_citations
        ) and (bool(cited_ids) or not required_citations)

    @staticmethod
    def _is_grounded_in_sources(answer, sources, allow_uncited_visual=False):
        stop_words = {
            "about", "after", "also", "are", "can", "cause", "causes", "does", "from", "have",
            "into", "is", "its", "may", "more", "than", "that", "the", "their", "there",
            "these", "this", "those", "was", "were", "what", "when", "which", "with", "will",
            "would", "your", "cattle", "cow", "cows", "disease", "diseases",
        }

        def terms(text):
            return {
                token for token in re.findall(r"[a-z0-9]+", text.casefold())
                if len(token) > 2 and token not in stop_words
            }

        source_by_id = {str(source["id"]): terms(source["excerpt"]) for source in sources}
        sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
        for sentence in sentences:
            clean_sentence = re.sub(r"\[\d+\]", "", sentence).strip(" -*#\t")
            if not clean_sentence:
                continue
            cited_ids = re.findall(r"\[(\d+)\]", sentence)
            if allow_uncited_visual and not cited_ids and DiseaseKnowledgeBase._is_visual_statement(clean_sentence):
                continue
            if not cited_ids:
                return False
            source_terms = set().union(*(source_by_id.get(source_id, set()) for source_id in cited_ids))
            if not (terms(clean_sentence) & source_terms):
                return False
        return True

    def ask(
        self,
        question,
        top_k=6,
        min_score=0.48,
        history=None,
        image_observations=None,
        prediction_context=None,
        image_provided=False,
    ):
        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("A question is required")
        if image_observations is not None and not isinstance(image_observations, str):
            raise ValueError("image_observations must be text")
        if prediction_context is not None and not isinstance(prediction_context, dict):
            raise ValueError("prediction_context must be an object")
        if not isinstance(image_provided, bool):
            raise ValueError("image_provided must be a boolean")
        if image_observations and len(image_observations) > 1000:
            raise ValueError("image_observations is too long")
        if prediction_context and len(json.dumps(prediction_context)) > 4000:
            raise ValueError("prediction_context is too large")
        validated_history = self._history_for_generation(history)
        previous_user_question = next(
            (message["content"] for message in reversed(validated_history) if message["role"] == "user"),
            "",
        )
        is_follow_up = bool(previous_user_question) and self._is_follow_up(normalized_question)
        scope_question = (
            f"{previous_user_question} {normalized_question}".strip()
            if is_follow_up
            else normalized_question
        )
        safety_note = (
            "For serious symptoms or suspected contagious disease, contact a qualified veterinarian "
            "and follow applicable local animal-health reporting guidance."
        )
        scope_context = " ".join(
            part for part in (scope_question, image_observations or "") if part
        )
        if not self._is_veterinary_question(scope_context):
            return {
                "question": normalized_question,
                "answer": "I can help with cattle-health questions using the available veterinary references. That question is outside my knowledge scope.",
                "answer_mode": "out_of_scope",
                "status": "out_of_scope",
                "passages": [],
                "sources": [],
                "safety_note": safety_note,
            }
        self._load_index()
        if not self._index_loaded or self._vectors is None:
            raise KnowledgeServiceError("The disease knowledge index has not been built")

        retrieval_question = " ".join(
            part for part in (scope_question, image_observations or "") if part
        )
        query_vector = self._encode([retrieval_question])[0]
        scores = self._vectors @ query_vector
        topic = self._topic_for_question(retrieval_question)
        best_indices = np.argsort(scores)[::-1][:min(max(int(top_k) * 5, 20), 40)]
        passages = []
        for index in best_indices:
            score = float(scores[index])
            if score < min_score:
                break
            chunk = self._chunks[int(index)]
            if topic and not self._matches_topic(chunk, topic):
                continue
            passages.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": round(score, 4),
            })
            if len(passages) >= max(1, min(int(top_k), 8)):
                break

        sources = self._build_sources(passages)
        if not passages:
            return {
                "question": normalized_question,
                "answer": "I couldn't find enough information about this in the available veterinary references.",
                "answer_mode": "no_evidence",
                "status": "no_evidence",
                "passages": [],
                "sources": [],
                "safety_note": safety_note,
            }
        api_key, model, _ = self._llm_configuration()
        if self._answer_generator is None and (not api_key or not model):
            return {
                "question": normalized_question,
                "answer": "AI answer generation is not configured yet. Please configure the language model to generate answers from the veterinary knowledge base.",
                "answer_mode": "llm_not_configured",
                "status": "llm_not_configured",
                "passages": passages,
                "sources": sources,
                "safety_note": safety_note,
            }
        answer, answer_mode, generation_warning = self._generate_answer(
            normalized_question,
            sources,
            validated_history,
            image_observations,
            prediction_context,
            image_provided,
        )
        result = {
            "question": normalized_question,
            "answer": answer,
            "answer_mode": answer_mode,
            "status": answer_mode,
            "passages": passages,
            "sources": sources,
            "safety_note": safety_note,
        }
        if image_observations:
            result["image_observations"] = image_observations
        if prediction_context:
            result["prediction_context"] = prediction_context
        if generation_warning:
            result["generation_warning"] = generation_warning
        return result