# Loan Approval Prediction System

An end-to-end machine-learning notebook that predicts whether a loan application will be approved. The project compares **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes** using a consistent train/test split and standard classification metrics.

> **Project focus:** data preparation, exploratory analysis, categorical encoding, feature scaling, feature engineering, and transparent model evaluation.

## Results at a glance

Evaluation uses an 80/20 train-test split (`random_state=42`) on 1,000 loan-application records. The positive class is **loan approved**.

| Model | Configuration | Accuracy | Precision | Recall | F1-score |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | Engineered features | **87.50%** | 79.03% | **80.33%** | **79.67%** |
| Gaussian Naive Bayes | Engineered features | 86.50% | **78.33%** | 77.05% | 77.69% |
| KNN | `k=5`, engineered features | 75.50% | 62.00% | 50.82% | 55.86% |
| Tuned KNN | Grid search (`k=9`, precision scoring) | 77.00% | 66.67% | 49.18% | 56.60% |

**Best overall model:** Logistic Regression achieved the highest accuracy, recall, and F1-score in the notebook's final comparison.

## Project structure

```text
Loan-Approval-System/
|-- credit_wise.ipynb          # Complete analysis, modelling, and evaluation
|-- loan_approval_data.csv     # Source dataset (1,000 records, 20 columns)
|-- requirements.txt           # Python dependencies
|-- .gitignore                 # Files excluded from version control
`-- README.md                  # Project overview and results
```

## Workflow

1. Load the loan-application dataset and inspect its structure.
2. Impute missing numeric values with the mean and categorical values with the mode.
3. Explore class balance and key applicant attributes through visualizations.
4. Remove the identifier field and encode categorical variables.
5. Split the data into training (80%) and test (20%) sets, then standardize features.
6. Train Logistic Regression, KNN, and Gaussian Naive Bayes classifiers.
7. Compare models with accuracy, precision, recall, F1-score, and confusion matrices.
8. Evaluate a feature-engineered variant using squared DTI-ratio and credit-score features.

## Dataset

The dataset contains 1,000 loan applications with applicant, financial, and property details. The source file has 50 missing values in each column; the notebook applies its existing imputation workflow before analysis and modelling. After that step, the target column (`Loan_Approved`) is distributed as follows:

| Class | Records | Share |
| --- | ---: | ---: |
| No | 702 | 70.2% |
| Yes | 298 | 29.8% |

Key inputs include applicant and co-applicant income, employment status, age, credit score, existing loans, debt-to-income ratio, savings, collateral value, loan details, and property information.

## Run locally

```bash
git clone <your-repository-url>
cd Loan-Approval-System
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
```

Run the notebook from top to bottom so every preprocessing and evaluation step is reproduced in order.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Notes

- The identifier column (`Applicant_ID`) is excluded before modelling.
- KNN hyperparameter search considers `k = 3, 5, 7, 9, 11` and selects the model using precision-based cross-validation.
- Results are specific to the included dataset and fixed split; this project is intended as a portfolio and learning project, not a production loan-decision system.
