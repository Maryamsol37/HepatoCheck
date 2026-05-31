# HepatoCheck

## Overview
HepatoCheck is a machine learning application for predicting Hepatitis C infection status based on clinical and demographic features.

## Quick Start
1. Install dependencies:
  ```
  pip install -r requirements.txt
  ```
2. Run the application:
  ```
  python main.py
  ```

## Project Structure
```
HepatoCheck/
├── main.py                    # Application entry point
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── data/                      # Data files
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── sample_inputs/         # Sample input files
├── models/                    # Trained models and artifacts
├── src/                       # Source code
│   ├── gui/                   # GUI components
│   ├── app/                   # Application controller
│   ├── ml/                    # Machine learning modules
│   ├── data/                  # Data processing modules
│   └── utils/                 # Utility functions
├── outputs/                   # Generated outputs
│   ├── reports/               # Generated reports
│   └── predictions/           # Prediction results
├── tests/                     # Test files
└── docs/                      # Documentation
```

## Features
HepatoCheck provides a user-facing Tkinter application for liver risk screening.

- Home page with project overview and navigation
- Single patient input form
- Input validation for missing or invalid clinical values
- Liver risk prediction using the trained machine learning model
- Result display showing risk classification, model confidence, probabilities, abnormal lab flags, recommendations, and top features
- Medical disclaimer page
- Batch CSV upload for multiple patient records
- Batch prediction table
- Export batch results to CSV
- Screening history page
- Export screening history to TXT
- Error handling for invalid inputs, missing files, and prediction issues

## Required Patient Inputs
For single-patient prediction and batch CSV prediction, the following input features are required:

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | Patient biological sex, accepted values: `m`, `f`, `male`, `female` |
| ALB | Albumin |
| ALP | Alkaline phosphatase |
| ALT | Alanine aminotransferase |
| AST | Aspartate aminotransferase |
| BIL | Bilirubin |
| CHE | Cholinesterase |
| CHOL | Cholesterol |
| CREA | Creatinine |
| GGT | Gamma-glutamyl transferase |
| PROT | Total protein |

## Batch CSV Format
The batch upload page accepts CSV files containing the same required patient features.

```csv
Age,Sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT
47,m,38.2,64,18,22,0.7,6.2,4.9,0.9,15,72
```

## Data
- Raw datasets: `data/raw/`
- Processed datasets: `data/processed/`
- Sample inputs: `data/sample_inputs/sample_patients.csv`

## Model Artifacts
Model artifacts are stored in `models/`.

| File | Description |
|------|------------|
| trained_model.pkl | Final RandomForest model |
| scaler.pkl | Feature scaler |
| feature_names.pkl | Ordered feature list |
| model_metrics.json | Evaluation metrics |
| best_pipeline.pkl | Full pipeline (SMOTE + RF) |
| advanced_metrics.json | Model comparison results |
| shap_pruned_pipeline.pkl | Reduced feature pipeline |

## Evaluation and Explainability
- Stratified train/test split, 5-fold cross validation, and nested cross validation for tuning
- ROC-AUC as the main metric with confusion matrix reporting
- SHAP for local explanations and global feature ranking

## Imbalance Handling
- SMOTE applied on training data
- Class weights used in RandomForest
- Threshold tuned to 0.2 instead of 0.5
- Calibration applied for probability stability

## Outputs
- Batch predictions: `outputs/predictions/`
- Reports: `outputs/reports/`

## Testing
Run the test suite:
```
python -m pytest
```

## Team Structure
- **ML Developer**: Data processing, ML models, training pipeline, model evaluation
- **GUI Developer**: Tkinter interface, application controller, input validation, result display, batch upload, export features
- **Shared**: Testing, documentation, utilities, final integration

## Contributing
See `docs/contribution_matrix.md` for contribution guidelines.
