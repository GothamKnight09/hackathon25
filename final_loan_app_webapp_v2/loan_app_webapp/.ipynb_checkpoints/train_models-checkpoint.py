import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import joblib

# Load data

data = pd.read_csv('database/customer_table.csv')
data['share_linkedin'] = data['share_linkedin'].apply( lambda x: 1 if x == 'Y' else 0 )
data['profession'] = data['profession'].apply( lambda x: 1 if x == 'Y' else 0 )

print(data)

# data = pd.get_dummies(data, columns=['profession'])

# Prepare features and target
X = data.drop(columns=['name', 'email'])
y = pd.read_csv('database/loan_application_table.csv')['loan_approved']

# Train models
lr_model = LinearRegression()
lr_model.fit(X, y)

lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X, y)

# Save models
joblib.dump(lr_model, 'models/linear_regression_model.pkl')
joblib.dump(lgbm_model, 'models/lgbm_model.pkl')
