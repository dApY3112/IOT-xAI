# Cyber-IOT Paper — XAI & Model Training

This repository contains code and artifacts used to train and analyze ML models for the Cyber-IOT paper, plus an explainability notebook. It includes a training script, saved model pipelines, and an interactive Jupyter notebook that demonstrates SHAP, LIME, PDP and a surrogate decision tree.

## Repository layout (key files)
- `train_models.py` — original training script (various models; LightGBM output seen in logs)
- `train_test_network.csv` — dataset used for training/evaluation (not ignored by default)
- `rf_pipeline.joblib`, `rf_pipeline_no_id.joblib` — saved sklearn pipelines (RandomForest) produced by notebook/scripts
- `xai_feature_importance.ipynb` — interactive notebook (SHAP, LIME, Captum, PDP, surrogate)
- `feature_importances_best_model.csv`, `permutation_importance_no_id.csv` — example output CSVs
- `.gitignore` — ignores caches, model artifacts and common outputs

> NOTE: Large model/artifact files (joblib, .npy, exported CSVs/PNGs) are included in `.gitignore` to avoid accidentally committing heavy files. Remove or edit `.gitignore` if you intend to track these.

## Quick prerequisites
- Python 3.10 (this repo was tested with Anaconda `py310` environment)
- Recommended packages (conda/pip):
  - numpy, pandas, scikit-learn, matplotlib, seaborn, joblib
  - lightgbm (optional), shap, lime
  - torch, captum (optional, for Integrated Gradients demo)

Example (PowerShell) pip commands:

```powershell
pip install numpy pandas scikit-learn matplotlib seaborn joblib
pip install lightgbm shap lime
# If using GPU and matching PyTorch CUDA build (choose right index for your CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install captum
```

Or create a conda environment (recommended):

```powershell
conda create -n cyberiot python=3.10 -y
conda activate cyberiot
pip install -r requirements.txt  # if you create one
```

## Running the notebook (interactive)
1. Start JupyterLab or Jupyter Notebook from the repository root (PowerShell):

```powershell
jupyter lab
# or
jupyter notebook
```

2. Open `xai_feature_importance.ipynb` and run the cells. The notebook is written to be guarded — missing optional packages (SHAP, LIME, captum) will be reported but will not break other sections.

## Running the notebook non-interactively
You can execute the notebook end-to-end using `nbconvert` or `papermill`. This is useful when you want to produce outputs without manually clicking through.

Example (PowerShell):

```powershell
# Using nbconvert (may require increased timeout for long cells)
jupyter nbconvert --to notebook --execute xai_feature_importance.ipynb --ExecutePreprocessor.timeout=600 --output xai_feature_importance.executed.ipynb

# Or use papermill to parameterize and run
pip install papermill
papermill xai_feature_importance.ipynb xai_feature_importance.run.ipynb
```

## Reproducing the baseline RF pipeline
- The notebook contains a cell to train a RandomForest pipeline and save it to `rf_pipeline.joblib` (and a no-id variant `rf_pipeline_no_id.joblib`) — run the training cell.
- If you prefer running a script, use `train_models.py` (inspect the code and modify paths as needed).

## Output artifacts
Typical artifacts the notebook may produce (saved to repository root unless changed):
- `shap_feature_importances_sample.csv`, `shap_aggregate_importances.csv`, `shap_values_sample.npy`
- `lime_explanation_sample.csv`
- `feature_importances_best_model.csv`, `permutation_importance_no_id.csv`
- `surrogate_tree.png`, `pdp_top3.png`
- `rf_pipeline.joblib`, `rf_pipeline_no_id.joblib`

## Notes & Troubleshooting
- SHAP shape mismatch: different SHAP versions and explainers may return arrays with different shapes (per-class × feature). The notebook includes robust extraction logic to handle common variants; if you see shape errors, try updating `shap` or re-run the SHAP cell.
- PDP errors on categorical features: PartialDependenceDisplay expects numeric columns or features mapped into the transformed numeric space. The notebook contains a PDP cell that maps features to the preprocessed space; use that cell when you see errors like "can't multiply sequence by non-int of type 'float'".
- If using CUDA for PyTorch/Captum, ensure installed torch build matches your GPU drivers and CUDA version.

## Committing to GitHub
- The repository includes a `.gitignore`. Before pushing, double-check which artifacts you want to track. Example push sequence (PowerShell):

```powershell
git init
git add .
git commit -m "Initial commit: training & XAI notebook"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Contact / Author
If you want me to run the notebook for you in this workspace (generate CSVs/plots), say "Run notebook" and I'll execute the relevant cells and report the saved artifacts.

---
README generated automatically. Adjust sections (requirements, dataset names) to match your preferences before publishing.
