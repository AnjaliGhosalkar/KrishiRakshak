# Disease Knowledge Assistant

The assistant reads PDF files from `disease_documents/`, extracts text with page numbers, and stores normalized semantic vectors and source metadata in the ignored `disease_knowledge/` folder. The index is reused for questions and can be refreshed from the assistant's **Reference library settings** panel.

## Unified text and image questions

The chat uses one `POST /knowledge/ask` endpoint for both modes. Text-only requests can use JSON; image questions use `multipart/form-data` with the question, optional image, recent conversation, and optional current prediction context. Image uploads must be decodable JPEG, PNG, WEBP, or BMP files no larger than 12 MB or 20 megapixels.

When an image is attached, the local BLIP model receives that image and the user's exact question. Its output is only a candidate visual observation: empty, generic, irrelevant, diagnostic, treatment-related, and unsupported injury claims are discarded. The grounded answer still comes from relevant PDF passages and a configured language model. If no reliable visual observation is available, the assistant says so rather than filling in an answer. The classifier prediction, when available for the same dashboard image, is labeled as uncertain context and is never treated as a diagnosis. Legacy `/vqa` remains available and keeps its `{question, answer}` response on successful analysis, but rejects unusable model outputs.

Available PDFs in this workspace:

- `cattle-diseases-farmers-guide.pdf`
- `foot-and-mouth-disease.pdf`
- `Lumpy-Skin-Disease.pdf`
- `Mastitis.pdf`
- `Mastitis2.pdf`
- `Srinivasetal2020LSD.pdf`
- `VeterinaryMedicineAndScience1.pdf`

Install the knowledge and local image-analysis dependencies in the same Python environment used to run Flask:

```powershell
python -m pip install -r requirements-knowledge.txt
python -m pip install -r requirements-vqa.txt
```

The first index build downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. It needs network access once. The first image question also downloads `Salesforce/blip-vqa-base` if it is not cached. PDFs with no extractable text (including scanned image-only PDFs) are skipped; OCR is not currently included.

## Required answer generation

The assistant uses the OpenAI-compatible Chat Completions API and the model named in `OPENAI_MODEL`. There is currently no model configured in this workspace. Until you configure one, cattle-health questions receive a setup message; retrieved PDF sentences are never presented as answers. The project does not call a paid service unless you configure one.

This project uses the OpenAI-compatible Chat Completions API. Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the same PowerShell window before starting Flask. `OPENAI_BASE_URL` is optional; it defaults to `https://api.openai.com/v1` and can point to another compatible provider or a local server.

```powershell
$env:OPENAI_API_KEY = "your-provider-api-key"
$env:OPENAI_MODEL = "your-model-name"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
python .\app.py
```

The app sends the question and retrieved, topic-filtered, page-tagged evidence to `{OPENAI_BASE_URL}/chat/completions`. The model must cite retrieved source IDs. Answers with invalid citations or no grounding overlap are rejected; the app never replaces them with copied PDF sentences. Keep the API key in the backend terminal only; never put it in frontend files. A local OpenAI-compatible service can use an empty key if it does not require authentication.

Start the backend and frontend in separate PowerShell terminals from the project directory:

```powershell
cd "$HOME\OneDrive\Desktop\KrishiRakshak\KrishiRakshak"
$env:OPENAI_API_KEY = "your-provider-api-key"
$env:OPENAI_MODEL = "your-model-name"
# For another OpenAI-compatible provider, also set OPENAI_BASE_URL to its /v1 URL.
python .\app.py
```

```powershell
cd "$HOME\OneDrive\Desktop\KrishiRakshak\KrishiRakshak\frontend"
npm install
npm run dev
```

The backend also needs the project's existing TensorFlow, scikit-learn, OpenCV, and model dependencies installed. If no key/model is configured, the endpoint returns `status: "llm_not_configured"`; this is a configuration error, not an LLM-generated answer.

Recent conversation turns are sent with follow-up questions while the page remains open. Conversation history is not saved after a page reload. The assistant is decision support, not a veterinary diagnosis; contact a qualified veterinarian for serious symptoms or suspected contagious disease and follow applicable local reporting guidance.
