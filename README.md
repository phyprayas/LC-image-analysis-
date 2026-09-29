# QDLC POM Image Analysis — Python Code Archive

GitHub-ready archive of the final/recommended Python scripts developed for the NaOH-treated CPCl/CPC:glycerol POM study.

## Sample naming
- **QDLC1 = 30:70**
- **QDLC2 = 50:50**

## Calibration and fixed analysis settings
- 50× POM images.
- Scale: **10 µm / 157 px = 0.0636942675 µm px⁻¹**.
- OrientationJ grid spacing: 20 px where used.
- Coherency threshold: C ≥ 0.30 where used.
- `S2D = |<exp(i 2 theta)>|`.
- `C2(r) = <cos[2(theta(x+r)-theta(x))]>`.
- Recovery model: `S(r) = S_inf - A exp(-r/lambda_S)`.

## Folder structure
### 01_Core_Analysis
Main mask/orientation/order/defect/NN/coupling/recovery/domain-comparison pipeline. `STEP10...V7.py` is the preferred final master-comparison script among the available Step-10 versions.

### 02_Psi_Orientation_Study
Texture-derived optical orientation Ψ, gradient/distortion, defect-centered analysis, winding validation, radial order/distortion, intensity, integrated comparison, representative FOV selection, and reliable-only winding analysis.

**Terminology:** Ψ is an image-derived/local optical texture orientation field, not a direct molecular director measurement.

### 03_Texture_T1_T2_T3
T1/T2/T3 ROI selection and texture-specific intensity/orientation/radial/azimuthal/core-boundary analyses. `TEXTURE_STUDY_STEP1_MULTI_IMAGE_ROI_SELECTION.py` allows T1/T2 from QDLC1 FOV4 and T3 from QDLC2 FOV3 (paths are editable).

### 04_Publication_Figures
Publication-ready comparison plots: global/local S2D, C2(r), defect density, unlike-sign nearest-neighbor fraction vs shuffled null, local order vs defect distance, order recovery + lambda_S, opposite-sign pair correlation, and entropy/domain-area comparisons.

### 05_Utilities
Conversion/recreation/file-picker/helper utilities.

## Important scientific cautions
- Winding detections should be described as **topological defect candidates** unless independently validated.
- T1/T2/T3 are **operational optical texture classes/populations**, not automatically thermodynamic phases.
- `1/2 |grad Psi|^2` is an orientation-distortion proxy, not direct Frank elastic-energy density unless an elastic constant is known.
- FOVs are spatial fields of view. If they come from one material preparation, do not describe them as independent material replicates.
- Pair-correlation/null and recovery results should be interpreted as image-level spatial statistics.

## How to use
1. Install packages: `pip install -r requirements.txt`
2. Open the desired script.
3. Edit Windows input/output paths at the top of the script.
4. Run in VS Code, Spyder, or Jupyter.
5. Keep raw inputs separate from generated output folders.

## Files deliberately not promoted as final
Superseded `_FIXED_V3`, `_FIXED_V4`, `_V5`, `_V6`, earlier diagnostic/test scripts, and the abandoned T1/T2/T3 ideal-topology/simulated-POM analysis were not placed in the main GitHub package. This keeps the repository focused on the final/recommended workflow.

## Note on 50:50 masking
The later 50:50 workflow used crystal/bright-inclusion exclusion before quantitative comparison. The available crystal-exclusion helper is included in `01_Core_Analysis`; preserve the same retained-texture logic when rerunning the dataset.
