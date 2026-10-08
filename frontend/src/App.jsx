import axios from 'axios'
import { useRef, useState } from 'react'
import './App.css'
import './Dashboard.css'
import BiomarkerDetails from './components/BiomarkerDetails'
import ClinicalSummary from './components/ClinicalSummary'
import EvidenceExplanation from './components/EvidenceExplanation'
import RankedEvidence from './components/RankedEvidence'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [vqaQuestion, setVqaQuestion] = useState('')
  const [vqaLoading, setVqaLoading] = useState(false)
  const [vqaResult, setVqaResult] = useState(null)
  const [vqaError, setVqaError] = useState(null)
  const [resultImageErrors, setResultImageErrors] = useState({})
  
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      setImagePreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      setVqaQuestion('')
      setVqaResult(null)
      setVqaError(null)
      setResultImageErrors({})
    }
  }

  const handlePredict = async () => {
    if (!imageFile) return
    
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('image', imageFile)
    
    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.error || "Failed to connect to the prediction server.")
    } finally {
      setLoading(false)
    }
  }

  const handleAskQuestion = async (event) => {
    event.preventDefault()
    if (!imageFile || !vqaQuestion.trim()) return

    setVqaLoading(true)
    setVqaError(null)
    setVqaResult(null)

    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('question', vqaQuestion)

    try {
      const response = await axios.post('http://localhost:5000/vqa', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setVqaResult(response.data)
    } catch (err) {
      console.error(err)
      setVqaError(err.response?.data?.error || 'Failed to get an answer from the VQA service.')
    } finally {
      setVqaLoading(false)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setImageFile(file)
      setImagePreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
      setVqaQuestion('')
      setVqaResult(null)
      setVqaError(null)
      setResultImageErrors({})
    }
  }

  const stageKey = (result?.stage || 'unknown').toLowerCase()
  const hasEbdrePanel =
    result &&
    (result.clinical_support_score != null ||
      result.evidence_strength ||
      result.ranked_evidence ||
      result.clinical_summary ||
      result.biomarkers ||
      result.ebdre_error)

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand-lockup" href="#top" aria-label="KrishiRakshak home">
          <span className="brand-mark" aria-hidden="true">KR</span>
          <span className="brand-name">KrishiRakshak<span>ANIMAL HEALTH REVIEW</span></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#prediction">Assessment</a>
          <a href="#clinical-support">Clinical support</a>
          <a href="#visual-evidence">Image evidence</a>
          <a href="#vqa">Ask about image</a>
        </nav>
        <div className="workspace-status"><span /> Image review workspace</div>
      </header>

      <main id="top" className="dashboard-main">
        <section className="welcome-row">
          <div>
            <p className="eyebrow"><span className="eyebrow-mark" /> LIVESTOCK IMAGE REVIEW</p>
            <h1>Field insight, grounded in the image.</h1>
            <p className="welcome-copy">Review a livestock image with prediction, evidence, and visual question answering in one workspace.</p>
          </div>
          <div className="review-file" aria-live="polite">
            <span className="review-file-label">CURRENT IMAGE</span>
            <span className="review-file-name">{imageFile?.name || 'No image selected'}</span>
            {imageFile && <span className="review-file-size">{(imageFile.size / (1024 * 1024)).toFixed(2)} MB</span>}
          </div>
        </section>

        <section className="upload-section" aria-labelledby="upload-title">
          <div
            className={`upload-zone ${imagePreview ? 'has-image' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            role="button"
            tabIndex={0}
            aria-label={imageFile ? 'Choose a different livestock image' : 'Choose a livestock image'}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />

            {imagePreview ? (
              <>
                <img src={imagePreview} alt="Selected livestock image preview" className="preview-image" />
                <span className="preview-change">Choose another image</span>
              </>
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon" aria-hidden="true"><span /></span>
                <strong>Drop an image to begin</strong>
                <p>or choose a photo from your device</p>
                <span className="upload-format">JPG · PNG · WEBP</span>
              </div>
            )}
          </div>

          <div className="upload-aside">
            <div className="upload-aside-heading">
              <span className="step-number">01</span>
              <div>
                <p className="section-kicker">START A REVIEW</p>
                <h2 id="upload-title">Add a field image</h2>
              </div>
            </div>
            <p className="upload-guidance">Choose a clear, well-lit photo of the area you want to review. Your selected image will be used for the prediction and any visual questions below.</p>
            <button
              className="predict-btn"
              onClick={handlePredict}
              disabled={!imageFile || loading}
            >
              {loading ? <><span className="spinner" /> Analyzing image</> : 'Run image assessment'}
            </button>
            <p className="privacy-note"><span aria-hidden="true">●</span> Image-based decision support. Confirm findings with a veterinary professional.</p>
            {error && <div className="error-message" role="alert">{error}</div>}
          </div>
        </section>

        {result && (
          <section className="results-section fade-in">
            <section id="prediction" className="result-card" aria-labelledby="prediction-title">
              <div className="result-header">
                <div>
                  <p className="section-kicker">MODEL OUTPUT</p>
                  <h2 id="prediction-title">Disease prediction</h2>
                </div>
                <div className="confidence-badge"><span>MODEL CONFIDENCE</span>{result.confidence ?? '—'}</div>
              </div>

              <div className="prediction-summary">
                <div className="disease-result">
                  <span className="result-label">PREDICTED CONDITION</span>
                  <h3 className="value disease">{result.disease ?? '—'}</h3>
                </div>
                <div className="stage-result">
                  <span className="result-label">PREDICTED STAGE</span>
                  <span className={`stage-tag stage-${stageKey}`}>{result.stage ?? '—'}</span>
                </div>
              </div>

              <div className="prediction-guidance-grid">
                <div className={`alert-box alert-${stageKey}`}>
                  <span className="guidance-label">ASSESSMENT NOTE</span>
                  <p>{result.alert ?? '—'}</p>
                </div>
                <div className="recommendation-box">
                  <span className="guidance-label">NEXT STEP</span>
                  <p>{result.recommendation ?? '—'}</p>
                  {result.description && <p className="description">{result.description}</p>}
                </div>
              </div>
            </section>

            {hasEbdrePanel && (
              <section id="clinical-support" className="ebdre-section" aria-labelledby="clinical-support-title">
                <div className="section-heading">
                  <div>
                    <p className="section-kicker">SEPARATE EVIDENCE LAYER</p>
                    <h2 id="clinical-support-title" className="ebdre-section-title">EBDRE clinical support</h2>
                  </div>
                  <p>Evidence summaries and image-derived measurements are shown separately from model confidence.</p>
                </div>
                <div className="clinical-grid">
                  <EvidenceExplanation result={result} />
                  <RankedEvidence rankedEvidence={result.ranked_evidence} />
                  <ClinicalSummary
                    summary={result.clinical_summary}
                    fallbackDisease={result.disease}
                    fallbackStage={result.stage}
                  />
                </div>
              </section>
            )}

            <section id="biomarkers" className="biomarker-section" aria-labelledby="biomarker-title">
              <div className="section-heading">
                <div>
                  <p className="section-kicker">IMAGE-DERIVED MEASUREMENTS</p>
                  <h2 id="biomarker-title" className="ebdre-section-title">Biomarker analysis</h2>
                </div>
                <p>Visual measurements are experimental and are not a diagnosis.</p>
              </div>
              <div className="biomarker-layout">
                <article className="ebdre-card biomarker-image-card">
                  <div className="panel-heading">
                    <h3>Biomarker image</h3>
                    <span className="panel-index">01</span>
                  </div>
                  {result.biomarker_image_url && !resultImageErrors.biomarker ? (
                    <img
                      className="result-visualization-image"
                      src={result.biomarker_image_url}
                      alt="Original image, biomarker mask overlay, and detected contours"
                      onError={() => setResultImageErrors((current) => ({ ...current, biomarker: true }))}
                    />
                  ) : (
                    <p className="ebdre-muted image-unavailable">Biomarker visualization is unavailable for this image.</p>
                  )}
                </article>
                <BiomarkerDetails biomarkers={result.biomarkers} />
              </div>
            </section>

            <section id="visual-evidence" className="ebdre-section" aria-labelledby="explainable-ai-title">
              <div className="section-heading">
                <div>
                  <p className="section-kicker">MODEL INTERPRETABILITY</p>
                  <h2 id="explainable-ai-title" className="ebdre-section-title">Image evidence</h2>
                </div>
                <p>Interpretability views are presented with the model each one explains.</p>
              </div>
              <div className="result-visualizations">
                <article className="ebdre-card result-visualization-card">
                  <div className="panel-heading"><h3>SHAP explanation</h3><span className="panel-index">02</span></div>
                  <p className="ebdre-subtitle">Feature attribution for the production Random Forest prediction.</p>
                  {result.xai?.random_forest_shap?.image_url && !resultImageErrors.shap ? (
                    <img
                      className="result-visualization-image"
                      src={result.xai.random_forest_shap.image_url}
                      alt={`SHAP visualization for ${result.xai.random_forest_shap.class}`}
                      onError={() => setResultImageErrors((current) => ({ ...current, shap: true }))}
                    />
                  ) : (
                    <p className="ebdre-muted">SHAP visualization is unavailable for this prediction.</p>
                  )}
                </article>

                <article className="ebdre-card result-visualization-card">
                  <div className="panel-heading"><h3>Grad-CAM visualization</h3><span className="panel-index">03</span></div>
                  <p className="ebdre-subtitle">
                    Explains the saved MobileNet classifier, not the production Random Forest prediction.
                  </p>
                  {result.xai?.saved_mobilenet_gradcam?.image_url && !resultImageErrors.gradcam ? (
                    <img
                      className="result-visualization-image"
                      src={result.xai.saved_mobilenet_gradcam.image_url}
                      alt={`Grad-CAM visualization for the MobileNet classifier, predicted ${result.xai.saved_mobilenet_gradcam.class}`}
                      onError={() => setResultImageErrors((current) => ({ ...current, gradcam: true }))}
                    />
                  ) : (
                    <p className="ebdre-muted">Grad-CAM visualization is unavailable for this prediction.</p>
                  )}
                </article>
              </div>
            </section>

            <section className="progression-view ebdre-card" aria-labelledby="progression-title">
              <div className="progression-heading">
                <div>
                  <h3 id="progression-title">Disease Progression Reference</h3>
                  <p className="ebdre-subtitle">Reference examples for the predicted disease across its stages.</p>
                </div>
                <span className={`progression-disease stage-${stageKey}`}>{result.disease}</span>
              </div>
              {['early', 'moderate', 'severe'].some((stage) => result.progression_images?.[stage]) ? (
                <div className="progression-grid">
                  {['early', 'moderate', 'severe'].map(stage => (
                    result.progression_images?.[stage] ? (
                      <div key={stage} className="progression-card">
                        <img src={result.progression_images[stage]} alt={`${stage} stage`} />
                        <div className={`stage-label stage-${stage}`}>{stage.charAt(0).toUpperCase() + stage.slice(1)} Stage</div>
                      </div>
                    ) : null
                  ))}
                </div>
              ) : (
                <p className="progression-empty">
                  {stageKey === 'none'
                    ? 'A Healthy result has no disease progression stages.'
                    : `No reference images are available for ${result.disease} progression.`}
                </p>
              )}
            </section>
          </section>
        )}

        {imageFile && (
          <section id="vqa" className="vqa-section ebdre-card" aria-labelledby="vqa-title">
            <div className="vqa-heading">
              <div className="vqa-symbol" aria-hidden="true">Q</div>
              <div>
                <p className="section-kicker">IMAGE-BASED ASSISTANT</p>
                <h2 id="vqa-title">Ask about this image</h2>
                <p className="ebdre-subtitle">BLIP answers your question from the uploaded image. This assistant does not diagnose disease.</p>
              </div>
            </div>
            <form className="vqa-form" onSubmit={handleAskQuestion}>
              <label htmlFor="vqa-question">Your question</label>
              <div className="vqa-controls">
                <input
                  id="vqa-question"
                  type="text"
                  value={vqaQuestion}
                  onChange={(event) => setVqaQuestion(event.target.value)}
                  placeholder="Ask anything about visible details..."
                  disabled={vqaLoading}
                />
                <button
                  type="submit"
                  className="vqa-ask-button"
                  disabled={!vqaQuestion.trim() || vqaLoading}
                >
                  {vqaLoading ? <><span className="spinner" /> Thinking</> : 'Ask image'}
                </button>
              </div>
            </form>
            {vqaError && <div className="error-message" role="alert">{vqaError}</div>}
            {vqaLoading && <p className="vqa-loading" role="status">BLIP is considering the image and your question...</p>}
            {vqaResult && (
              <div className="vqa-answer" aria-live="polite">
                <p className="vqa-question-echo">{vqaResult.question}</p>
                <p className="vqa-answer-text">{vqaResult.answer}</p>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
