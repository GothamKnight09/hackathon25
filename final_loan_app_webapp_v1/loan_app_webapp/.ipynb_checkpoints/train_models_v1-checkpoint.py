import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load datasets
def load_data():
    customers = pd.read_csv('database/customer_table.csv')
    applications = pd.read_csv('database/loan_application_table.csv')
    payments = pd.read_csv('database/payment_history_table.csv')
    credit = pd.read_csv('database/credit_bureau_table.csv')
    behavior = pd.read_csv('database/credit_bureau_behavior_data.csv')
    return customers, applications, payments, credit, behavior

# Preprocessing pipeline
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='drop')
    return preprocessor

# Train and return all models
def train_models():
    customers, applications, payments, credit, behavior = load_data()

    # Merge for LightGBM model
    merged_all = applications.merge(customers, on='customer_id', how='left')
    merged_all = merged_all.merge(credit, on='customer_id', how='left')
    merged_all = merged_all.merge(behavior, on=['customer_id', 'credit_report_id'], how='left')

    # Merge for Linear Regression models (only approved)
    approved_apps = applications[applications['status'] == 'Approved']
    approved_merged = approved_apps.merge(customers, on='customer_id', how='left')
    approved_merged = approved_merged.merge(credit, on='customer_id', how='left')

    # Define features/targets
    X_lr = approved_merged.drop(columns=['application_name', 'product_id', 'application_date', 'loan_amount', 'loan_term_months', 'purpose', 'status', 'rejection_reason', 'interest_rate', 'customer_id'])
    y_amt = approved_merged['loan_amount']
    y_rate = approved_merged['interest_rate']

    X_lgb = merged_all.drop(columns=['application_name', 'product_id', 'application_date', 'loan_amount', 'loan_term_months', 'purpose', 'status', 'rejection_reason', 'interest_rate', 'customer_id'])
    y_lgb = merged_all['status'].apply(lambda x: 1 if x == 'Approved' else 0)

    # Build preprocessor
    preprocessor_lr = build_preprocessor(X_lr)
    preprocessor_lgb = build_preprocessor(X_lgb)

    # Define pipelines
    lr_amt_model = Pipeline([
        ('preprocessor', preprocessor_lr),
        ('regressor', LinearRegression())
    ])

    lr_rate_model = Pipeline([
        ('preprocessor', preprocessor_lr),
        ('regressor', LinearRegression())
    ])

    lgbm_class_model = Pipeline([
        ('preprocessor', preprocessor_lgb),
        ('classifier', lgb.LGBMClassifier())
    ])

    lgbm_amt_model = Pipeline([
        ('preprocessor', preprocessor_lr),
        ('regressor', lgb.LGBMRegressor())
    ])

    lgbm_rate_model = Pipeline([
        ('preprocessor', preprocessor_lr),
        ('regressor', lgb.LGBMRegressor())
    ])

    # Fit models
    lr_amt_model.fit(X_lr, y_amt)
    lr_rate_model.fit(X_lr, y_rate)
    lgbm_class_model.fit(X_lgb, y_lgb)
    lgbm_amt_model.fit(X_lr, y_amt)
    lgbm_rate_model.fit(X_lr, y_rate)

    return lr_amt_model, lr_rate_model, lgbm_class_model, lgbm_amt_model, lgbm_rate_model

# Save all models
def save_models():
    lr_amt, lr_rate, lgbm_class, lgbm_amt, lgbm_rate = train_models()
    joblib.dump(lr_amt, 'models/linear_regression_loan_amount.pkl')
    joblib.dump(lr_rate, 'models/linear_regression_interest_rate.pkl')
    joblib.dump(lgbm_class, 'models/lgbm_model.pkl')
    joblib.dump(lgbm_amt, 'models/lgbm_regression_loan_amount.pkl')
    joblib.dump(lgbm_rate, 'models/lgbm_regression_interest_rate.pkl')
    print("All models saved in 'models/' folder")

# Run saving
if __name__ == '__main__':
    save_models()
