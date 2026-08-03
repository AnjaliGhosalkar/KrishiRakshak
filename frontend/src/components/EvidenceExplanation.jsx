import { formatLabel, formatScore } from './ebdreFormat'

/**
 * AI Evidence Explanation — CSS, evidence strength, confidence level.
 */
function EvidenceExplanation({ result }) {
  if (!result) return null

  const css = formatScore(result.clinical_support_score)
  const strength = result.evidence_strength
  const confidenceLevel = result.confidence_level
  const cnnConfidence = formatScore(result.cnn_confidence) ?? result.confidence

  const hasEbdre =
    css != null ||
    (strength != null && strength !== '') ||
    (confidenceLevel != null && confidenceLevel !== '')

  if (!hasEbdre && result.ebdre_error) {
    return (
      <div className="ebdre-card">
        <h3>AI Evidence Explanation</h3>
        <p className="ebdre-muted">Evidence analysis unavailable for this prediction.</p>
      </div>
    )
  }

  if (!hasEbdre) return null

  return (
    <div className="ebdre-card">
      <h3>AI Evidence Explanation</h3>
      <p className="ebdre-subtitle">
        How strongly visual biomarkers support the prediction (separate from CNN confidence).
      </p>

      <div className="ebdre-metrics">
        <div className="ebdre-metric">
          <span className="ebdre-metric-label">Clinical Support Score</span>
          <span className="ebdre-metric-value">
            {css != null ? `${css}` : '—'}
            {css != null && <span className="ebdre-metric-unit">/ 100</span>}
          </span>
        </div>
        <div className="ebdre-metric">
          <span className="ebdre-metric-label">Evidence Strength</span>
          <span className={`ebdre-pill strength-${(strength || 'unknown').toLowerCase()}`}>
            {formatLabel(strength)}
          </span>
        </div>
        <div className="ebdre-metric">
          <span className="ebdre-metric-label">Confidence Level</span>
          <span className={`ebdre-pill confidence-${(confidenceLevel || 'unknown').toLowerCase()}`}>
            {formatLabel(confidenceLevel)}
          </span>
        </div>
        {cnnConfidence != null && (
          <div className="ebdre-metric">
            <span className="ebdre-metric-label">CNN Confidence</span>
            <span className="ebdre-metric-value subtle">
              {String(cnnConfidence).includes('%') ? cnnConfidence : `${cnnConfidence}%`}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export default EvidenceExplanation
