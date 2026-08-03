/**
 * Format API support keys like "strong_support" → "Strong Support"
 */
export function formatLabel(value) {
  if (value == null || value === '') return '—'
  return String(value)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export function formatBiomarkerName(key) {
  const names = {
    texture: 'Texture',
    texture_roughness: 'Texture',
    density: 'Lesion Density',
    lesion_density: 'Lesion Density',
    color: 'Color Variation',
    color_variation: 'Color Variation',
    clustering: 'Lesion Clustering',
    lesion_clustering: 'Lesion Clustering',
    boundary: 'Boundary Irregularity',
    boundary_irregularity: 'Boundary Irregularity',
  }
  return names[key] || formatLabel(key)
}

export function formatScore(value) {
  if (value == null || Number.isNaN(Number(value))) return null
  const n = Number(value)
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}

export function formatBiomarkerValue(value) {
  if (value == null || value === '') return 'Unavailable'
  const n = Number(value)
  if (Number.isNaN(n)) return String(value)
  return n.toFixed(4)
}
