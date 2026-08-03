# Experimental Visual Biomarker Extraction

Research prototype for KrishiRakshak EBDRE. These are **image-derived visual statistics**, not medical diagnoses.

## Quick start

```bash
# Single image
python test_biomarkers.py test_images/udder1.jpg

# With debug visualization
python test_biomarkers.py dataset/lumpy/moderate/100_jpg.rf.5126a763111433d3c5366ec9e4840e8d.jpg --debug biomarker_debug

# Batch CSV export
python test_biomarkers.py --dir dataset/lumpy/moderate --csv biomarker_results.csv
```

## Module layout

- `extractor.py` — main `extract_biomarkers()` interface
- `lesion_detection.py` — modular candidate lesion mask (swappable later)
- `lesion_density.py`, `texture.py`, `color_variation.py`, `clustering.py`, `boundary.py`
- `debug_visualization.py` — optional segmentation debug panels

## Python API

```python
from biomarker_extraction import extract_biomarkers

result = extract_biomarkers("path/to/image.jpg", save_debug=True, debug_output_dir="biomarker_debug")
print(result)
```

## Dependencies

Uses existing project libraries: OpenCV, NumPy, Pillow. No new packages were added.
