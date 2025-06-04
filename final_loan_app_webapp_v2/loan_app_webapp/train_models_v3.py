import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import os

# -------------------
# Load Data
# -------------------
def load_data():
    customer = pd.read_csv("database/customers.csv")
    credit = pd.read_csv("database/credit_bureau_data.csv")
    loan_app = pd.read_csv("database/loan_applications.csv")
    payment = pd.read_csv("database/payment_history.csv")
    behavior = pd.read_csv("database/credit_bureau_behavior_data.csv")
    return customer, credit, loan_app, payment, behavior

# -------------------
# Prepare Linear Regression and Logistic Data (Approved only)
# -------------------
def prepare_data_for_lr(customer, credit, loan_app):
    df = loan_app.copy()
    df = df.merge(customer, on='customer_id')
    df = df.merge(credit, on='customer_id')

    df['loan_amount'] = df['loan_amount'].astype(float)
    df['interest_rate'] = df['interest_rate'].astype(float)
    df['approved'] = df['status'].apply(lambda x: 1 if x == 'Approved' else 0)

    features = [
        'annual_income', 'employment_length_years', 'debt_to_income_ratio',
        'credit_score', 'num_accounts', 'num_inquiries_6m'
    ]
    X = df[features]
    y_loan = df['loan_amount']
    y_interest = df['interest_rate']
    y_approval = df['approved']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_loan, y_interest, y_approval, scaler

# -------------------
# Train Linear and Logistic Models
# -------------------
def train_linear_models(X_scaled, y_loan, y_interest, y_approval):
    lr_loan = LinearRegression().fit(X_scaled, y_loan)
    lr_int = LinearRegression().fit(X_scaled, y_interest)
    lr_approval = LogisticRegression().fit(X_scaled, y_approval)
    return lr_loan, lr_int, lr_approval

# -------------------
# Prepare LGBM Data (All applications + behavior)
# -------------------
def prepare_data_for_lgbm(customer, credit, loan_app, behavior):
    df = loan_app.merge(customer, on="customer_id")
    df = df.merge(credit, on="customer_id")
    df = df.merge(behavior, on="customer_id", how='left')

    df['approved'] = df['status'].apply(lambda x: 1 if x == 'Approved' else 0)

    ordinal_map = {'low': 1, 'medium': 2, 'high': 3}

    df['linkedin_activity_level'] = (
        df['linkedin_activity_level']
        .map(ordinal_map)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(2)
        .astype(int)
    )

    df['twitter_activity_level'] = (
        df['twitter_activity_level']
        .map(ordinal_map)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(2)
        .astype(int)
    )

    categorical_cols = ['employment_status', 'gender', 'housing_status']
    numeric_cols = [
        'annual_income', 'employment_length_years', 'debt_to_income_ratio',
        'credit_score', 'num_accounts', 'num_inquiries_6m',
        'bankruptcy_flag', 'foreclosure_flag'
    ]
    behavior_cols = [
        'linkedin_activity_level', 'twitter_activity_level', 'facebook_posts_last_6m',
        'youtube_avg_views', 'youtube_monetized'
    ]

    X = df[numeric_cols + categorical_cols + behavior_cols]
    y_approval = df['approved']
    y_loan = df['loan_amount'].astype(float)
    y_interest = df['interest_rate'].astype(float)

    woe_encoder = ce.WOEEncoder(cols=categorical_cols)
    X_encoded = woe_encoder.fit_transform(X, y_approval)

    return X_encoded, y_approval, y_loan, y_interest, woe_encoder

# -------------------
# Train LGBM Models
# -------------------
def train_lgbm_models(X, y_approval, y_loan, y_interest):
    lgbm_approval = lgb.LGBMClassifier(random_state=42)
    lgbm_approval.fit(X, y_approval)

    lgbm_loan = lgb.LGBMRegressor(random_state=42)
    lgbm_loan.fit(X, y_loan)

    lgbm_interest = lgb.LGBMRegressor(random_state=42)
    lgbm_interest.fit(X, y_interest)

    return lgbm_approval, lgbm_loan, lgbm_interest

# -------------------
# Save Models
# -------------------
def save_models(lr_loan, lr_int, lr_approval, scaler, lgbm_approval, lgbm_loan, lgbm_interest, woe_encoder):
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_loan, "models/lr_loan_amount.pkl")
    joblib.dump(lr_int, "models/lr_interest_rate.pkl")
    joblib.dump(lr_approval, "models/lr_approval.pkl")
    joblib.dump(scaler, "models/lr_scaler.pkl")
    joblib.dump(lgbm_approval, "models/lgbm_approval.pkl")
    joblib.dump(lgbm_loan, "models/lgbm_loan_amount.pkl")
    joblib.dump(lgbm_interest, "models/lgbm_interest_rate.pkl")
    joblib.dump(woe_encoder, "models/woe_encoder.pkl")

# -------------------
# Run Training Pipeline
# -------------------
def main():
    customer, credit, loan_app, payment, behavior = load_data()

    X_lr, y_loan, y_int, y_approval, scaler = prepare_data_for_lr(customer, credit, loan_app)
    print(X_lr)
    lr_loan, lr_int, lr_approval = train_linear_models(X_lr, y_loan, y_int, y_approval)

    X_lgbm, y_lgbm_approval, y_lgbm_loan, y_lgbm_int, woe_encoder = prepare_data_for_lgbm(customer, credit, loan_app, behavior)
    print(X_lgbm.columns.to_list())
    lgbm_approval, lgbm_loan, lgbm_interest = train_lgbm_models(X_lgbm, y_lgbm_approval, y_lgbm_loan, y_lgbm_int)

    save_models(lr_loan, lr_int, lr_approval, scaler, lgbm_approval, lgbm_loan, lgbm_interest, woe_encoder)
    print("\nAll models trained and saved in 'models/' folder.")

if __name__ == '__main__':
    main()
