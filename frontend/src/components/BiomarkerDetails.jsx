import { formatBiomarkerName, formatBiomarkerValue } from './ebdreFormat'

const BIOMARKER_ORDER = [
  'texture_roughness',
  'lesion_density',
  'color_variation',
  'boundary_irregularity',
  'lesion_clustering',
]

/**
 * Numeric biomarker details when present in the API response.
 */
function BiomarkerDetails({ biomarkers }) {
  if (!biomarkers || typeof biomarkers !== 'object') return null

  let entries = BIOMARKER_ORDER
    .filter((key) => Object.prototype.hasOwnProperty.call(biomarkers, key))
    .map((key) => [key, biomarkers[key]])

  // Also show any unexpected short keys if full names are missing
  if (entries.length === 0) {
    const shortOrder = ['texture', 'density', 'color', 'boundary', 'clustering']
    entries = shortOrder
      .filter((key) => Object.prototype.hasOwnProperty.call(biomarkers, key))
      .map((key) => [key, biomarkers[key]])
  }

  if (entries.length === 0) return null

  return (
    <div className="ebdre-card">
      <h3>Biomarker Details</h3>
      <p className="ebdre-subtitle">Image-derived visual measurements (experimental).</p>

      <div className="biomarker-grid">
        {entries.map(([key, value]) => (
          <div key={key} className="biomarker-cell">
            <span className="biomarker-name">{formatBiomarkerName(key)}</span>
            <span className={`biomarker-value ${value == null ? 'unavailable' : ''}`}>
              {formatBiomarkerValue(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default BiomarkerDetails
