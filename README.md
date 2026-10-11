# CODSOFT Task 3: Customer Churn Prediction

A machine learning model that predicts whether a bank customer will leave (churn), built for the **CodSoft Machine Learning Internship**.

**Author:** Saheli | [GitHub](https://github.com/itsmesaheli2005) | [LinkedIn](https://www.linkedin.com/in/saheli-sarkar-35097a3a5)

## Results

Evaluated on a held-out, stratified 20% test set (2,000 customers). Churn is the positive class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 71.35% | 38.72% | 70.02% | 0.4987 | 0.7772 |
| Random Forest | 85.25% | 65.56% | 57.99% | 0.6154 | 0.8584 |
| **Gradient Boosting (best)** | **86.90%** | **78.21%** | 49.39% | 0.6054 | **0.8702** |

Models are ranked by ROC-AUC, which measures how well a model separates churners from loyal customers regardless of the decision threshold.

Best model, Gradient Boosting, on the 2,000 test customers:

- Caught **201 of 407** customers who churned (49.4% recall).
- Of the customers it flagged as churners, **78.2%** really did leave (precision).
- Wrongly flagged only **56 of 1,593** loyal customers.

**Trade-off:** Logistic Regression catches 70% of churners but flags many loyal customers by mistake. Gradient Boosting is more precise but misses about half of the churners at the default 0.5 threshold. For a retention campaign, lowering the threshold would catch more churners at the cost of more false alarms.

## What drives churn?

The most influential features in the best model were **Age**, **NumOfProducts**, **IsActiveMember**, **Balance** and **Geography (Germany)**. See `feature_importance.png`.

## How it works

1. **Load** the data (10,000 customers, 20.4% churned) and detect the target column (`Exited`).
2. **Drop identifier columns** (`RowNumber`, `CustomerId`, `Surname`), which carry no predictive signal.
3. **Preprocess:** scale numeric features, one-hot encode categorical features (`Geography`, `Gender`).
4. **Split** into 80% train and 20% test, stratified so both keep the same churn rate.
5. **Compare** Logistic Regression, Random Forest and Gradient Boosting, handling class imbalance with class weights where supported.
6. **Select** the model with the best ROC-AUC, plot the results, retrain it on all data and save it.

## Project structure

```
CODSOFT_3/
├── churn_predictor.py        # training + evaluation + scoring script
├── confusion_matrix.png      # confusion matrix on the test set
├── roc_curves.png            # ROC curves for all three models
├── feature_importance.png    # top features driving churn
├── requirements.txt
└── README.md
```

## Dataset

Bank customer churn dataset `Churn_Modelling` (linked in the CodSoft task sheet). Place the file in a `data/` folder:

```
data/
└── Churn_Modelling.csv
```

The dataset file is not included in this repo.

## How to run

```bash
pip install -r requirements.txt

# train, evaluate and save the model
python churn_predictor.py

# score a new file of customers (same columns, without the Exited column)
python churn_predictor.py --score data/new_customers.csv
```

Outputs are written to an `output/` folder: the trained model (`churn_model.joblib`) and the three plots.

It also runs in Google Colab with no installs: upload the script, create a `data/` folder containing the dataset, and run `!python churn_predictor.py`.

## Possible improvements

- Tune the decision threshold to trade precision for recall.
- Hyperparameter search (GridSearchCV or Optuna) on Gradient Boosting.
- Try XGBoost or LightGBM, and engineered features such as balance-to-salary ratio.
- Use SMOTE or other resampling to handle class imbalance.

## Tech stack

Python, pandas, scikit-learn, matplotlib, joblib

## Demo

LinkedIn video: *add your link here after posting* #codsoft
