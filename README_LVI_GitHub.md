# Breast cancer LVI prediction: analysis code

This repository contains the Python analysis pipeline for a two-center study developing and externally validating preoperative models for lymphovascular invasion (LVI) in breast cancer.

## Study design implemented in the code

- **Center 1:** model development, nested cross-validation, hyperparameter selection, and final model fitting.
- **Center 2:** locked external validation only. It is not used for model selection, feature selection, threshold optimization, or recalibration.
- **M1:** clinical features.
- **M2:** clinical features plus conventional MRI features.
- **M3:** M2 plus the prespecified breast peritumor score (BPS).
- Postoperative nodal status (`N`) is excluded from all preoperative LVI prediction models and is used only in exploratory DFS analyses.
- AUC confidence intervals in the performance table and ROC figure use the same bootstrap procedure and random seed (`RANDOM_SEED = 2026`).

## Repository contents

- `LVI_prediction_analysis_GitHub.py` — main analysis script.
- `requirements_LVI.txt` — Python dependencies.
- `.gitignore` — excludes local datasets and generated analysis outputs.

## Data privacy

**Patient-level data are not included in this repository.** Do not upload identifiable or confidential clinical data, source Excel workbooks, patient-level prediction files, or other restricted study materials to a public GitHub repository. Keep the input workbooks on an approved secure drive and provide access only in accordance with institutional and ethics requirements.

## Input files

The script expects the following files in a local data directory:

1. `Data1.xlsx` — Center 1 development cohort.
2. `Data2.xlsx` — Center 2 locked external validation cohort.
3. `agreement data.xlsx` — reader agreement data used by the reproducibility analysis.

The Excel column names must match those referenced in the script. In particular, the conventional MRI categorical feature must be named exactly **`Internal enhancement`** in both cohort workbooks and in the reader-agreement data where applicable. The former label `lesion type` is no longer used by this version of the script.

The script currently includes a cohort provenance check expecting Center 1 = 634 patients with 141 LVI-positive cases and Center 2 = 156 patients with 34 LVI-positive cases. This check is intentional for the study dataset version. Do not disable or change it unless the analysis dataset has been formally updated and the manuscript methods/results are revised accordingly.

## Installation

Use Python 3.10–3.12 in a clean virtual environment where possible.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements_LVI.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_LVI.txt
```

## Run the analysis

Set `LVI_DATA_DIR` to the local directory containing the three input workbooks. For example, in Windows Command Prompt:

```bat
set LVI_DATA_DIR=I:\article\test\LVI\use
python LVI_prediction_analysis_GitHub.py
```

In PowerShell:

```powershell
$env:LVI_DATA_DIR = "I:\article\test\LVI\use"
python .\LVI_prediction_analysis_GitHub.py
```

If `LVI_DATA_DIR` is not set, the script looks for the input workbooks in the same directory as the Python script.

Generated tables, figures, model files, and analysis configuration are written to `LVI_analysis_results` under the selected data directory.

## Reproducibility and interpretation notes

- The random seed is set to `2026`; the number of bootstrap replicates is set in the script.
- The nested cross-validation is row-level stratified CV. The study team must verify one row per patient. If multiple lesions or rows per patient are present, use a verified patient identifier and group-aware splitting before publication.
- The clinical decision threshold is not optimized automatically. `FIXED_THRESHOLD` remains `None` unless a threshold is independently justified and prespecified.
- DFS analyses are exploratory and should be interpreted in light of the small number of observed events.
- The script exports analysis outputs; it does not independently verify source-data accuracy, eligibility, ethics approval, or clinical interpretations.

## Before public release

1. Confirm that all input workbook column names and category coding match the final analysis dataset.
2. Confirm the cohort counts and event coding against the locked study database.
3. Run the script in a clean environment and archive the package versions and outputs used for the submitted manuscript.
4. Review the repository for patient-level data, identifiers, local paths, credentials, and unpublished confidential information before making it public.
5. Add the license required by the authors' institution or journal policy. No license is included by default.
