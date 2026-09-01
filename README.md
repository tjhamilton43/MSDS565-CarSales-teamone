# MSDS 565 Car Sales — Team1

End-to-end CRISP-DM regression project using the instructor-provided modified Car Sales dataset.

## Run order

1. `P1_CarSales-1_EDA_Team1.ipynb` profiles structure, missingness, distributions, relationships, and data-quality flags. It writes no downstream artifacts.
2. `P1_CarSales-2_Preprocessing_Team1.ipynb` cleans and engineers features without price leakage, preserves categorical metadata, performs the fixed 80/20 split, and writes `processed_train.csv` and `processed_test.csv`.
3. `P1_CarSales-3_Modeling_Team1.ipynb` tunes three sklearn regressors and compares three Keras architectures. It writes `model_metrics.csv` and full-feature models.
4. `P1_CarSales-4_FeatureSelection_Team1.ipynb` keeps the top 100 Random Forest features, applies iterative VIF filtering, retrains every model family, and writes selected splits, selected metrics, and `_selected` models.
5. `P1_CarSales-5_xAI_Team1.ipynb` loads the selected Random Forest, creates global/local SHAP explanations, and audits make, body type, and listing color with Fairlearn.

## Data

Place `CarSales_Small.csv` in `data/`, keep it in the class data folder shown in Notebook 1, or update `RAW_CANDIDATES`. Generated CSV files are intentionally ignored by Git because they can exceed repository limits and are reproducible from Notebook 2 onward.

## Reproduce

```bash
conda env create -f environment.yml
conda activate msds565-team1
jupyter lab
```

Execute notebooks in numeric order. All random splits and folds use `random_state=42`.
