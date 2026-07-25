import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      setImagePreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
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
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Livestock Disease Predictor</h1>
        <p>Upload an image to detect diseases, severity, and get actionable recommendations.</p>
      </header>

      <main className="main-content">
        <section className="upload-section">
          <div 
            className={`upload-zone ${imagePreview ? 'has-image' : ''}`}
            onClick={() => fileInputRef.current.click()}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="image/*" 
              style={{ display: 'none' }} 
            />
            
            {imagePreview ? (
              <img src={imagePreview} alt="Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📸</div>
                <p>Click or drag image here to upload</p>
              </div>
            )}
          </div>
          
          <button 
            className="predict-btn" 
            onClick={handlePredict} 
            disabled={!imageFile || loading}
          >
            {loading ? <span className="spinner"></span> : "Analyze Image"}
          </button>
          
          {error && <div className="error-message">{error}</div>}
        </section>

        {result && (
          <section className="results-section fade-in">
            <div className="result-card">
              <div className="result-header">
                <h2>Prediction Results</h2>
                <div className="confidence-badge">Confidence: {result.confidence}</div>
              </div>
              
              <div className="result-details">
                <div className="detail-item">
                  <span className="label">Disease:</span>
                  <span className="value disease">{result.disease}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Stage:</span>
                  <span className={`value stage stage-${result.stage.toLowerCase()}`}>
                    {result.stage}
                  </span>
                </div>
              </div>

              <div className={`alert-box alert-${result.stage.toLowerCase()}`}>
                <strong>Alert:</strong> {result.alert}
              </div>
              
              <div className="recommendation-box">
                <h3>Recommendation</h3>
                <p>{result.recommendation}</p>
                <p className="description">{result.description}</p>
              </div>
            </div>

            {result.progression_images && Object.keys(result.progression_images).length > 0 && (
              <div className="progression-view">
                <h3>Disease Progression Reference</h3>
                <div className="progression-grid">
                  {['early', 'moderate', 'severe'].map(stage => (
                    result.progression_images[stage] ? (
                      <div key={stage} className="progression-card">
                        <img src={result.progression_images[stage]} alt={`${stage} stage`} />
                        <div className={`stage-label stage-${stage}`}>{stage.charAt(0).toUpperCase() + stage.slice(1)} Stage</div>
                      </div>
                    ) : null
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
