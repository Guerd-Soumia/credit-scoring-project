# Credit Scoring – Random Forest Classifier

**Individual contribution: Agent 2 – Random Forest Classifier**

## Introduction

This repository documents my individual machine learning contribution: a Random
Forest classifier that estimates the risk of payment difficulties for loan
applicants from the Home Credit dataset. The work covers the full workflow I was
responsible for — preparing the borrower data, training the classifier, evaluating
it, analysing feature importance and decision thresholds, saving reusable model
artifacts, and providing a small test script.

## Project Objective

The objective is to predict a binary credit-risk outcome for each applicant:

- **0**: Applicant did not have recorded payment difficulties.
- **1**: Applicant had recorded payment difficulties.

The model outputs a probability of class `1`, interpreted as the applicant's
default risk. In the notebook, the class-1 counts are printed as "Default in
train/test" and `predict_proba(...)[:, 1]` is used as the risk score. The target
column is `TARGET`.

## Dataset

The dataset used here is the **Home Credit** borrower dataset
(`application_train.csv`). The notebook records the following structure:

| Item | Verified value |
| --- | --- |
| Raw dataset shape | 307,511 rows × 122 columns |
| Columns after dropping >50% missing values | 81 |
| Final feature count | 79 |
| Training rows | 246,008 |
| Test rows | 61,503 |
| Defaults in training set | 19,860 |
| Defaults in test set | 4,965 |

After removing the label `TARGET` and the identifier `SK_ID_CURR`, the remaining
79 columns are used as features. These include borrower attributes such as
`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `CNT_CHILDREN`, `CODE_GENDER`,
`FLAG_OWN_CAR`, and `FLAG_OWN_REALTY`.

The original CSV and the cleaned pickle are **not included in this Git repository**.
They are excluded by `.gitignore` because of their size. See **Dataset Setup** for
where to place them locally.

## Data Preprocessing

All preprocessing is implemented in `notebooks/02_clean.ipynb`:

1. Load `data/home_credit/application_train.csv`.
2. Drop columns where more than 50% of values are missing (41 columns dropped).
3. Replace the `DAYS_EMPLOYED` sentinel value `365243` with `NaN`; the notebook
   annotates this value as "unemployed".
4. Encode the 13 categorical (`object`) columns with `LabelEncoder`.
5. Fill all remaining missing values with the column median. The notebook reports
   **0** remaining missing values after this step.
6. Define `y = TARGET` and `X` as all columns except `TARGET` and `SK_ID_CURR`.
7. Split with `train_test_split(test_size=0.2, random_state=42, stratify=TARGET)`.
8. Fit `StandardScaler` on the training split and transform both splits.
9. Save `(X_train, X_test, y_train, y_test, feature_names)` to
   `data/clean/hc_data.pkl`, the fitted scaler to `models/hc_scaler.pkl`, and the
   ordered feature names to `models/hc_feature_names.pkl`.

The scaler is fitted only on training data; the test data is transformed with the
same fitted scaler.

## Random Forest Model

A Random Forest is an ensemble of decision trees. Each tree is trained on a
bootstrapped sample of the data and considers a random subset of features when
splitting. For classification, the final class is based on the votes (or averaged
class probabilities) of all trees. This makes the model more robust than a single
decision tree and well suited to tabular credit data with non-linear relationships.

`class_weight="balanced"` is used because only about 8% of applicants are in class
`1`; it increases the influence of the minority class during training.

The model configuration verified from the notebook and from the saved pickle is:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `n_estimators` | 100 | Number of decision trees. |
| `max_depth` | 10 | Maximum depth of each tree. |
| `class_weight` | `"balanced"` | Higher weight for the minority default class. |
| `random_state` | 42 | Reproducible sampling and training. |
| `n_jobs` | -1 | Use all available CPU cores. |

At inference time, raw applicant values are placed in a DataFrame with the saved
feature order, transformed by the fitted scaler, then passed to the model:

```python
raw = pd.DataFrame([row], columns=feature_names)
scaled = scaler.transform(raw)
probability_of_default = rf.predict_proba(scaled)[0][1]
```

## Training Workflow

The pipeline in `notebooks/04_agent2_rf.ipynb` is:

1. **Load prepared data** from `data/clean/hc_data.pkl`:
   `X_train, X_test, y_train, y_test, feature_names`.
2. **Prepare features and target**: the features are already separated from
   `TARGET` and scaled; `y_train` and `y_test` hold the binary labels.
3. **Use the existing split**: the 80/20 stratified split was created in
   `02_clean.ipynb` with `random_state=42` and is loaded already prepared.
4. **Train the Random Forest** using the parameters listed above.
5. **Generate predictions** with `predict()` for class labels and
   `predict_proba(...)[:, 1]` for default probabilities.
6. **Evaluate** with `classification_report`, `confusion_matrix`, and
   `roc_auc_score`; plot the confusion matrix and top-10 feature importances;
   print threshold comparisons.
7. **Save the model** to `models/agent2_rf.pkl`.

## Model Evaluation

The results below are the values printed in `04_agent2_rf.ipynb` on the held-out
test set of 61,503 applicants.

### Classification report

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| 0 — no recorded difficulties | 0.96 | 0.74 | 0.83 | 56,538 |
| 1 — recorded difficulties | 0.17 | 0.61 | 0.26 | 4,965 |
| Accuracy |  |  | 0.73 | 61,503 |
| Macro average | 0.56 | 0.67 | 0.55 | 61,503 |
| Weighted average | 0.89 | 0.73 | 0.79 | 61,503 |

- **ROC-AUC:** 0.7374947744330942

Meaning of the metrics in this context:

- **Precision (class 1):** of the applicants predicted as risky, how many truly
  had payment difficulties. The value 0.17 means many flagged applicants are false
  alarms.
- **Recall (class 1):** of all applicants who truly had difficulties, how many the
  model found. The value 0.61 means about 61% were detected.
- **F1-score:** the balance between precision and recall for a class.
- **ROC-AUC:** how well the model ranks a risky applicant above a non-risky one,
  independent of a single threshold.

### Confusion matrix

Verified from the saved model on the notebook's test split:

```text
                         Predicted 0   Predicted 1
Actual 0 (no difficulty)     41,691         14,847
Actual 1 (difficulty)         1,945          3,020
```

Interpretation: the model correctly identifies 3,020 of 4,965 applicants with
recorded difficulties and misses 1,945. It also flags 14,847 applicants who did
not have recorded difficulties. This reflects the recall-oriented trade-off from
`class_weight="balanced"`.

## Feature Importance

The notebook computes `rf_model.feature_importances_` and plots the top 10 using
a horizontal bar chart. Verified values from the saved model:

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | `EXT_SOURCE_3` | 0.209822 |
| 2 | `EXT_SOURCE_2` | 0.192981 |
| 3 | `DAYS_EMPLOYED` | 0.061203 |
| 4 | `DAYS_BIRTH` | 0.048387 |
| 5 | `AMT_GOODS_PRICE` | 0.033251 |
| 6 | `DAYS_LAST_PHONE_CHANGE` | 0.032479 |
| 7 | `NAME_EDUCATION_TYPE` | 0.031226 |
| 8 | `AMT_CREDIT` | 0.027677 |
| 9 | `DAYS_ID_PUBLISH` | 0.026683 |
| 10 | `AMT_ANNUITY` | 0.022846 |

Feature importance shows which columns the forest used most when splitting trees.
It helps describe the model's behaviour, but it does **not** prove that a feature
causes credit risk. These are impurity-based importance values, not causal effects.

## Threshold Analysis

`predict()` uses the default probability cutoff of **0.5**. The notebook also
tests alternative thresholds by applying `(y_proba_rf > t)` for `t` in
`[0.3, 0.4, 0.5, 0.6]`:

| Threshold on P(class 1) | Precision (class 1) | Recall (class 1) |
| --- | --- | --- |
| 0.3 | 0.10 | 0.96 |
| 0.4 | 0.12 | 0.83 |
| 0.5 | 0.17 | 0.61 |
| 0.6 | 0.24 | 0.34 |

Lowering the threshold catches more applicants who truly had difficulties but
increases false alarms. Raising it improves precision but misses more risky
applicants. Threshold selection matters because the cost of approving a risky
loan is different from the cost of rejecting a good applicant.

## Model Artifacts

| File | Purpose |
| --- | --- |
| `models/agent2_rf.pkl` | The trained `RandomForestClassifier` (100 trees, 79 input features). Loaded by the test script for predictions. |
| `models/hc_scaler.pkl` | The `StandardScaler` fitted during preprocessing. Raw inputs must be transformed with it before scoring. |
| `models/hc_feature_names.pkl` | The ordered list of 79 feature names, ensuring new input rows match the training feature structure. |

Together, these artifacts allow the model to be reused with compatible feature
processing.

## Project Structure

```text
random-forest-credit-scoring/
├── notebooks/
│   ├── 02_clean.ipynb
│   └── 04_agent2_rf.ipynb
├── models/
│   ├── agent2_rf.pkl
│   ├── hc_scaler.pkl
│   └── hc_feature_names.pkl
├── data/                         # local folders; datasets are ignored by Git
│   ├── home_credit/
│   │   └── application_train.csv  (downloaded locally)
│   └── clean/
│       └── hc_data.pkl            (downloaded locally)
├── test_agent2.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `notebooks/02_clean.ipynb`: preprocessing and artifact creation.
- `notebooks/04_agent2_rf.ipynb`: training, evaluation and model saving.
- `models/`: reusable trained artifacts.
- `data/`: local folders for the downloaded datasets; no dataset files are tracked
  by Git.
- `test_agent2.py`: artifact and pipeline smoke test.

## Installation

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

The artifacts were created with Python 3.10.22 and scikit-learn 1.7.2; Python
3.10–3.12 is recommended for exact reproduction.

## Dataset Setup

Place the files exactly here:

```text
data/
├── home_credit/
│   └── application_train.csv
└── clean/
    └── hc_data.pkl
```

- `application_train.csv` is needed only if you want to regenerate the cleaned
  data by running `02_clean.ipynb`.
- `hc_data.pkl` is needed by `04_agent2_rf.ipynb` and by the optional real-row
  check in `test_agent2.py`.

Neither file is committed to this repository. Create the folders, then download
the files locally from the published dataset locations:

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path data\home_credit, data\clean
curl.exe -L -o data/home_credit/application_train.csv https://media.githubusercontent.com/media/Guerd-Soumia/credit-scoring-project/main/data/home_credit/application_train.csv
curl.exe -L -o data/clean/hc_data.pkl https://media.githubusercontent.com/media/Guerd-Soumia/credit-scoring-project/main/data/clean/hc_data.pkl
```

```bash
# Linux / macOS
mkdir -p data/home_credit data/clean
curl -L -o data/home_credit/application_train.csv https://media.githubusercontent.com/media/Guerd-Soumia/credit-scoring-project/main/data/home_credit/application_train.csv
curl -L -o data/clean/hc_data.pkl https://media.githubusercontent.com/media/Guerd-Soumia/credit-scoring-project/main/data/clean/hc_data.pkl
```

If the cleaned dataset is missing, run `notebooks/02_clean.ipynb` after placing
`application_train.csv`; it recreates `hc_data.pkl`, `hc_scaler.pkl`, and
`hc_feature_names.pkl`.

## How to Run

1. Install the dependencies as shown above.
2. Prepare the dataset in the `data/` folders as shown above.
3. If preprocessing is required, open `notebooks/02_clean.ipynb` in VS Code or
   Jupyter.
4. Run its cells in order; confirm it prints shapes `(307511, 122)`, `(307511, 81)`,
   `79` features, and train/test shapes `(246008, 79)` / `(61503, 79)`.
5. Open `notebooks/04_agent2_rf.ipynb`.
6. Run its cells in order to train, evaluate, plot, analyse thresholds and save the
   model.
7. Review the classification report, ROC-AUC, confusion matrix, feature-importance
   chart and threshold output.

Both notebooks determine the project root from the working directory, so they work
whether Jupyter starts at the repository root or in `notebooks/`.

> Running `04_agent2_rf.ipynb` writes over `models/agent2_rf.pkl`, as the notebook
> did when producing the recorded results. Keep a copy if you need to preserve the
> committed artifact.

## Testing

Run:

```bash
python test_agent2.py
```

The script verifies that:

- the model, scaler and feature-name files exist and load;
- the model and scaler are the expected types;
- all three agree on **79 features**;
- expected borrower columns are present in the feature-name list;
- two synthetic applicant rows can be scaled and scored;
- the risky synthetic profile receives a higher default probability than the
  safer one;
- if `data/clean/hc_data.pkl` is present, a real held-out row survives an
  `inverse_transform` → `transform` scaler round-trip and receives the same
  probability as direct scoring.

It does not re-train the model, recompute the notebook metrics, or test the
preprocessing notebook.

## Limitations

- The default class is only about 8% of the data; recall is prioritised through
  `class_weight="balanced"`, which lowers precision.
- At the 0.5 threshold, many non-difficulty applicants are flagged. Threshold
  choice must reflect the relative cost of false alarms versus missed risky
  applicants.
- Results come from one historical dataset and may not generalise to other
  populations or time periods.
- Predictions should not be the sole basis for real lending decisions; they are a
  risk signal, not a final decision.
- Feature importance is based on impurity reduction in the forest. It describes
  model usage, not causation, and can be biased toward features with many possible
  split values.
- Pickle artifacts are version-sensitive; the saved files were created with
  scikit-learn 1.7.2.
- The test's synthetic profiles set only 7 of 79 features; they verify pipeline
  compatibility, not realistic applicant scoring.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Conclusion

This repository presents my individual Random Forest implementation for credit
risk: preprocessing the Home Credit borrower data, training a 100-tree balanced
forest, evaluating it on a held-out test set, analysing feature importance and
decision thresholds, saving reusable artifacts, and verifying the saved pipeline
with a small test script.
