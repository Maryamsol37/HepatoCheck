# HepatoCheck

## Overview
HepatoCheck is a machine learning application for predicting Hepatitis C infection status based on clinical and demographic features.

## Project Structure
```
HepatoCheck/
├── main.py                    # Application entry point
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── data/                      # Data files
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   └── sample_inputs/        # Sample input files
├── models/                   # Trained models and artifacts
├── src/                      # Source code
│   ├── gui/                  # GUI components
│   ├── app/                  # Application controller
│   ├── ml/                   # Machine learning modules
│   ├── data/                 # Data processing modules
│   └── utils/                # Utility functions
├── outputs/                  # Generated outputs
│   ├── reports/             # Generated reports
│   └── predictions/         # Prediction results
├── tests/                    # Test files
└── docs/                     # Documentation
```

## Installation
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the application:
```
python main.py
```

## Application Features

HepatoCheck provides a user-facing Tkinter application for liver risk screening.

Main features include:

- Home page with project overview and navigation
- Single patient input form
- Input validation for missing or invalid clinical values
- Liver risk prediction using the trained machine learning model
- Result display page showing:
  - Risk classification
  - Model confidence
  - Low-risk and possible-risk probabilities
  - Abnormal lab value flags
  - Recommendation message
  - Top important model features
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

The CSV file must include the following columns:

```csv
Age,Sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT




## Team Structure
- **ML Developer**: Data processing, ML models, training pipeline, model evaluation
- **GUI Developer**: Tkinter interface, application controller, input validation, result display, batch upload, export features
- **Shared**: Testing, documentation, utilities, final integration

## Contributing
See `docs/contribution_matrix.md` for contribution guidelines.


---

# 📦 Model Artifacts

| File | Description |
|------|------------|
| trained_model.pkl | Final RandomForest model |
| scaler.pkl | Feature scaler |
| feature_names.pkl | Ordered feature list |
| model_metrics.json | Evaluation metrics |
| best_pipeline.pkl | Full pipeline (SMOTE + RF) |
| advanced_metrics.json | Model comparison results |
| shap_pruned_pipeline.pkl | Reduced feature pipeline |

---

# 🧪 Evaluation Strategy

- Stratified train/test split
- 5-Fold Cross Validation
- Nested Cross Validation for tuning
- ROC-AUC used as main metric
- Confusion matrix for classification performance

---

# 🔍 Explainability

We used SHAP (SHapley Additive Explanations) to:

- Explain individual predictions
- Rank feature importance
- Visualize global model behavior


---

# 🚨 Imbalance Handling

Because dataset is highly imbalanced:

- SMOTE applied on training data
- Class weights used in RandomForest
- Threshold tuned to 0.2 instead of 0.5
- Calibration applied for probability stability

---

# 🖥️ GUI System

The system includes:

### Input Form
- Patient data entry
- Validation & preprocessing

### Result View
- Risk label
- Confidence score
- Abnormal markers
- Top contributing features

---

# 📌 Key Outcome

HepatoCheck provides:
- High accuracy liver disease prediction
- Interpretable ML decisions
- Clinically meaningful outputs

---

# 👨‍💻 Technologies Used

- Python
- Scikit-learn
- Pandas / NumPy
- SHAP
- LightGBM
- CatBoost
- Imbalanced-learn (SMOTE)
- Tkinter (GUI)

---

# 📈 Performance Summary

- Best Model: RandomForest
- ROC-AUC: 0.9878
- Calibrated ROC-AUC: 0.9975
- Stable generalization across folds

---

