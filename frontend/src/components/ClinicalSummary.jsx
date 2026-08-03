/**
 * Clinical Evidence Summary from EBDRE reporting output.
 */
function ClinicalSummary({ summary, fallbackDisease, fallbackStage }) {
  if (!summary || typeof summary !== 'object') {
    return (
      <div className="ebdre-card">
        <h3>Clinical Evidence Summary</h3>
        <p className="ebdre-muted">Clinical summary not available for this prediction.</p>
      </div>
    )
  }

  const title = summary.title || 'Clinical Evidence Summary'
  const prediction = summary.prediction || fallbackDisease || '—'
  const stage = summary.stage || fallbackStage || '—'
  const supporting = Array.isArray(summary.supporting_evidence)
    ? summary.supporting_evidence
    : []
  const unavailable = Array.isArray(summary.unavailable_evidence)
    ? summary.unavailable_evidence
    : []
  const conflicting = Array.isArray(summary.conflicting_evidence)
    ? summary.conflicting_evidence
    : []
  const overall = summary.overall_statement

  return (
    <div className="ebdre-card">
      <h3>{title}</h3>

      <div className="summary-meta">
        <div className="detail-item">
          <span className="label">Prediction</span>
          <span className="value">{prediction}</span>
        </div>
        <div className="detail-item">
          <span className="label">Stage</span>
          <span className="value">{stage}</span>
        </div>
      </div>

      {supporting.length > 0 && (
        <div className="summary-block">
          <h4>Supporting Evidence</h4>
          <ul className="summary-list">
            {supporting.map((line, i) => (
              <li key={`support-${i}`}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {unavailable.length > 0 && (
        <div className="summary-block">
          <h4>Unavailable Evidence</h4>
          <ul className="summary-list muted">
            {unavailable.map((line, i) => (
              <li key={`unavail-${i}`}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {conflicting.length > 0 && (
        <div className="summary-block">
          <h4>Conflicting Evidence</h4>
          <ul className="summary-list conflict">
            {conflicting.map((line, i) => (
              <li key={`conflict-${i}`}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {overall && (
        <div className="summary-overall">
          <strong>Overall:</strong> {overall}
        </div>
      )}
    </div>
  )
}

export default ClinicalSummary
