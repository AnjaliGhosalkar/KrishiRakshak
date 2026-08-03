import { formatBiomarkerName, formatLabel } from './ebdreFormat'

/**
 * Ranked Evidence list from EBDRE ranking output.
 */
function RankedEvidence({ rankedEvidence }) {
  const items = Array.isArray(rankedEvidence) ? rankedEvidence : []

  if (items.length === 0) {
    return (
      <div className="ebdre-card">
        <h3>Ranked Evidence</h3>
        <p className="ebdre-muted">No ranked evidence available for this prediction.</p>
      </div>
    )
  }

  return (
    <div className="ebdre-card">
      <h3>Ranked Evidence</h3>
      <p className="ebdre-subtitle">Visual cues ordered by contribution to the prediction.</p>

      <ul className="ranked-evidence-list">
        {items.map((item, index) => {
          const biomarker = item?.biomarker
          const support = item?.support
          const importance = item?.importance
          return (
            <li key={`${biomarker || 'item'}-${index}`} className="ranked-evidence-item">
              <div className="ranked-evidence-main">
                <span className="ranked-rank">#{index + 1}</span>
                <span className="ranked-name">{formatBiomarkerName(biomarker)}</span>
              </div>
              <div className="ranked-evidence-tags">
                <span className={`ebdre-pill support-${(support || 'unknown').toLowerCase()}`}>
                  {formatLabel(support)}
                </span>
                <span className={`ebdre-pill importance-${(importance || 'unknown').toLowerCase()}`}>
                  {formatLabel(importance)}
                </span>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default RankedEvidence
