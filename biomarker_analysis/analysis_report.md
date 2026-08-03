# KrishiRakshak Biomarker Validation Report

Experimental analysis of image-derived visual biomarkers. Not medical validation.

## 1. Dataset Used
- Root: `D:\cow_skin_detection\dataset`
- This is the dataset used by the current project (`app.py`, training scripts).
- **Foot and Mouth mapping:** `dataset/foot/` and `dataset/mouth/` are preserved as separate folders.
  Grouped disease analysis uses the label **Foot and Mouth** for both.

## 2. Dataset Size
- Total images in dataset: 8699
- Images processed in this run: 120
- Successful extractions: 120
- Failed extractions: 0

### Images by dataset folder
- `foot` -> Foot and Mouth: 86
- `healthy` -> Healthy: 2948
- `lumpy` -> Lumpy: 3724
- `mastitis` -> Mastitis: 474
- `mouth` -> Foot and Mouth: 1467

## 3. Sampling Method
- Mode: **sampled dataset**
- `--max-per-class`: 30 per mapped disease label
- Random seed: 42

## 4. Biomarker Extraction Method
Existing `biomarker_extraction/` package (unchanged):
- Lesion Density: lesion pixel ratio
- Texture Roughness: uniform LBP + Sobel gradient
- Color Variation: HSV S/V standard deviation
- Lesion Clustering: connected-component centroid distances
- Boundary Irregularity: contour circularity deviation

## 5. Data Quality Results
- **lesion_density**: mean=0.1622, missing=0.0%
- **texture_roughness**: mean=0.4661, missing=0.0%
- **color_variation**: mean=0.7757, missing=0.0%
- **lesion_clustering**: mean=0.4193, missing=49.17%
- **boundary_irregularity**: mean=0.6671, missing=38.33%

## 6. Disease-level Statistics
See `disease_statistics.csv` for full tables.

### lesion_density
- **Healthy** (n=30): mean=0.24, median=0.0072, std=0.3346, min=0.0, max=0.9532
- **Lumpy** (n=30): mean=0.1657, median=0.004, std=0.294, min=0.0, max=0.9622
- **Mastitis** (n=30): mean=0.1428, median=0.0025, std=0.2833, min=0.0, max=0.858
- **Foot and Mouth** (n=30): mean=0.1003, median=0.0036, std=0.2204, min=0.0, max=0.8311

### texture_roughness
- **Healthy** (n=30): mean=0.4929, median=0.5091, std=0.0625, min=0.3598, max=0.5788
- **Lumpy** (n=30): mean=0.5087, median=0.52, std=0.0512, min=0.3568, max=0.5814
- **Mastitis** (n=30): mean=0.4045, median=0.4096, std=0.0669, min=0.2774, max=0.5191
- **Foot and Mouth** (n=30): mean=0.4582, median=0.4808, std=0.0663, min=0.3209, max=0.5476

### color_variation
- **Healthy** (n=30): mean=0.7672, median=0.7942, std=0.1258, min=0.4054, max=0.9615
- **Lumpy** (n=30): mean=0.7823, median=0.8001, std=0.1298, min=0.568, max=1.0
- **Mastitis** (n=30): mean=0.7383, median=0.7582, std=0.1597, min=0.3855, max=1.0
- **Foot and Mouth** (n=30): mean=0.8149, median=0.8329, std=0.1239, min=0.5115, max=1.0

### lesion_clustering
- **Healthy** (n=14): mean=0.3883, median=0.4133, std=0.1505, min=0.0601, max=0.6212
- **Lumpy** (n=17): mean=0.4859, median=0.5091, std=0.2491, min=0.0, max=0.8424
- **Mastitis** (n=15): mean=0.3982, median=0.402, std=0.1584, min=0.2052, max=0.7321
- **Foot and Mouth** (n=15): mean=0.3937, median=0.4112, std=0.1573, min=0.1159, max=0.651

### boundary_irregularity
- **Healthy** (n=16): mean=0.7036, median=0.7626, std=0.2011, min=0.2586, max=0.9145
- **Lumpy** (n=21): mean=0.6636, median=0.6655, std=0.1608, min=0.3539, max=0.919
- **Mastitis** (n=18): mean=0.6134, median=0.5988, std=0.193, min=0.242, max=0.8613
- **Foot and Mouth** (n=19): mean=0.6911, median=0.7185, std=0.1591, min=0.416, max=0.9531

## 7. Stage-level Statistics
See `stage_statistics.csv`. Stage analysis preserves `dataset_folder`.

## 8. Missing-value Analysis
- lesion_density: 0.0% unavailable
- texture_roughness: 0.0% unavailable
- color_variation: 0.0% unavailable
- lesion_clustering: 49.17% unavailable
- boundary_irregularity: 38.33% unavailable

## 9. Distribution Observations
- Box plots saved under `plots/disease_boxplots/` and `plots/stage_boxplots/`.
- Disease mean heatmap saved under `plots/mean_comparisons/`.
- These plots are exploratory only; no causal claims are made.

## 10. Statistical Tests
- **lesion_density** | Kruskal-Wallis | Across all diseases | stat=0.8564 | p=0.8359
- **lesion_density** | One-way ANOVA | Across all diseases | stat=1.2564 | p=0.2927
- **lesion_density** | Mann-Whitney U | Healthy vs Lumpy | stat=448.5 | p=0.9879
- **lesion_density** | Mann-Whitney U | Healthy vs Mastitis | stat=486.0 | p=0.5858
- **lesion_density** | Mann-Whitney U | Healthy vs Foot and Mouth | stat=493.5 | p=0.5111
- **texture_roughness** | Kruskal-Wallis | Across all diseases | stat=36.268 | p=6.572e-08
- **texture_roughness** | One-way ANOVA | Across all diseases | stat=16.5908 | p=4.908e-09
- **texture_roughness** | Mann-Whitney U | Healthy vs Lumpy | stat=383.0 | p=0.3255
- **texture_roughness** | Mann-Whitney U | Healthy vs Mastitis | stat=750.0 | p=9.511e-06
- **texture_roughness** | Mann-Whitney U | Healthy vs Foot and Mouth | stat=593.0 | p=0.03514
- **color_variation** | Kruskal-Wallis | Across all diseases | stat=3.8164 | p=0.282
- **color_variation** | One-way ANOVA | Across all diseases | stat=1.6622 | p=0.179
- **color_variation** | Mann-Whitney U | Healthy vs Lumpy | stat=427.0 | p=0.7394
- **color_variation** | Mann-Whitney U | Healthy vs Mastitis | stat=485.0 | p=0.61
- **color_variation** | Mann-Whitney U | Healthy vs Foot and Mouth | stat=351.5 | p=0.1474
- **lesion_clustering** | Kruskal-Wallis | Across all diseases | stat=1.8208 | p=0.6104
- **lesion_clustering** | One-way ANOVA | Across all diseases | stat=1.0093 | p=0.3954
- **lesion_clustering** | Mann-Whitney U | Healthy vs Lumpy | stat=93.0 | p=0.3114
- **lesion_clustering** | Mann-Whitney U | Healthy vs Mastitis | stat=111.0 | p=0.8103
- **lesion_clustering** | Mann-Whitney U | Healthy vs Foot and Mouth | stat=105.5 | p=1

## 11. Effect-size Observations
- lesion_density Healthy vs Lumpy: Cohen's d=0.2359
- lesion_density Healthy vs Mastitis: Cohen's d=0.3135
- lesion_density Healthy vs Foot and Mouth: Cohen's d=0.4933
- texture_roughness Healthy vs Lumpy: Cohen's d=-0.2763
- texture_roughness Healthy vs Mastitis: Cohen's d=1.3653
- texture_roughness Healthy vs Foot and Mouth: Cohen's d=0.5371
- color_variation Healthy vs Lumpy: Cohen's d=-0.1181
- color_variation Healthy vs Mastitis: Cohen's d=0.2013
- color_variation Healthy vs Foot and Mouth: Cohen's d=-0.3822
- lesion_clustering Healthy vs Lumpy: Cohen's d=-0.4632
- lesion_clustering Healthy vs Mastitis: Cohen's d=-0.0639
- lesion_clustering Healthy vs Foot and Mouth: Cohen's d=-0.0346
- boundary_irregularity Healthy vs Lumpy: Cohen's d=0.2235
- boundary_irregularity Healthy vs Mastitis: Cohen's d=0.4581
- boundary_irregularity Healthy vs Foot and Mouth: Cohen's d=0.0696

## 12. Biomarker Correlation
See `correlation_matrix.csv` and `plots/correlation/biomarker_correlation_heatmap.png`.


## 13. Potential Strengths
- texture_roughness

## 14. Potential Weaknesses
- lesion_clustering: Currently unreliable (missing=49.2%; Kruskal-Wallis p=0.6104; max |Cohen's d|=0.463)
- boundary_irregularity: Currently unreliable (missing=38.3%; Kruskal-Wallis p=0.4186; max |Cohen's d|=0.458)

## 15. Limitations
- Segmentation may respond to background, fur, shadows, and lighting.
- Color variation is scene-level and not disease-specific by itself.
- Clustering/boundary biomarkers may be unavailable when too few lesion regions are detected.
- foot and mouth are separate folders grouped only for analysis.
- Exact duplicate images may exist; they were not removed.
- This run used sampled data; full-dataset statistics may differ.

## 16. Recommendations for Next Research Step
1. Review debug visualizations for segmentation quality on each disease folder.
2. Refine lesion detection before creating disease-specific biomarker profiles.
3. Proceed to Step 3: disease/stage-specific biomarker profiles using evidence from this report.
4. Do not assign CSS weights until profiles are validated on held-out data.

## Optional Baseline Experiment
- Performed: yes
- Accuracy: 0.4583
- Macro F1: 0.4631
- See `optional_baseline/metrics.txt`

## Biomarker Assessment Table

| Biomarker | Missing % | KW p-value | Max |d| vs Healthy | Recommendation |
|-----------|-----------|------------|---------------------|----------------|
| lesion_density | 0.0 | 0.8359338849158656 | 0.4933 | Needs further validation |
| texture_roughness | 0.0 | 6.572303564968563e-08 | 1.3653 | Potentially useful |
| color_variation | 0.0 | 0.28198296868907163 | 0.3822 | Needs further validation |
| lesion_clustering | 49.17 | 0.6104172417542537 | 0.4632 | Currently unreliable |
| boundary_irregularity | 38.33 | 0.41855975549808566 | 0.4581 | Currently unreliable |