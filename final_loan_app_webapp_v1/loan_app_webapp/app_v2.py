# Flask app logic for Loan Application System
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os
import webbrowser
import uuid

app = Flask(__name__)
app.secret_key = 'batman_secret_key'

# Load models and encoders
lr_loan = joblib.load("models/lr_loan_amount.pkl")
lr_int = joblib.load("models/lr_interest_rate.pkl")
lr_approval = joblib.load("models/lr_approval.pkl")
scaler = joblib.load("models/lr_scaler.pkl")
lgbm_model = joblib.load("models/lgbm_model.pkl")
woe_encoder = joblib.load("models/woe_encoder.pkl")
lgbm_loan_amount = joblib.load("models/lgbm_loan_amount.pkl")
lgbm_interest_rate = joblib.load("models/lgbm_interest_rate.pkl")



# Routes
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        credentials = pd.read_csv('database/credentials.csv')
        user = credentials[(credentials['username'] == username) & (credentials['password'] == password)]

        if not user.empty:
            session['username'] = username
            return redirect(url_for('loan_form'))
        else:
            return render_template('register.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_user = {
            "username": request.form['username'],
            "password": request.form['password']
        }
        new_user_df = pd.DataFrame([new_user])
        new_user_df.to_csv('database/credentials.csv', mode='a', index=False, header=False)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/loan_form', methods=['GET', 'POST'])
def loan_form():
    if request.method == 'POST':
        customer_id = str(uuid.uuid4())
        application_id = str(uuid.uuid4())

        first_name = request.form['first_name']
        last_name = request.form['last_name']


        # # Fetch bureau data
        # bureau_data = pd.read_csv("database/customer_bureau.csv")
        # bureau_row = bureau_data[
        #     (bureau_data['first_name'].str.lower() == first_name.lower()) &
        #     (bureau_data['last_name'].str.lower() == last_name.lower())
        # ]

        # if bureau_row.empty:
        #     return f"No bureau data found for {first_name} {last_name}"

        # bureau_row = bureau_row.iloc[0]  # Get the matching row

        form_data = {
            'customer_id': customer_id,
            'application_id': application_id,
            'first_name': first_name,
            'last_name': last_name,
            'dob': request.form['dob'],
            'email': request.form['email'],
            'phone': request.form['phone'],
            'address': request.form['address'],
            # 'gender': request.form['gender'],
            'employment_status': request.form['employment_status'],
            'employment_length_years': int(request.form['employment_length_years']),
            'housing_status': request.form['housing_status'],
            'annual_income': float(request.form['income']),
            # 'debt_to_income_ratio': float(request.form['debt_to_income_ratio']),
            'credit_score': int(request.form['credit_score']),
            # 'num_accounts': int(request.form['num_accounts']),
            # 'num_inquiries_6m': int(request.form['num_inquiries_6m']),
            # 'bankruptcy_flag': int(request.form['bankruptcy_flag']),
            # 'foreclosure_flag': int(request.form['foreclosure_flag']),
            # 'linkedin_activity_level': request.form['linkedin_activity_level'],
            # 'twitter_activity_level': request.form['twitter_activity_level'],
            # 'facebook_posts_last_6m': int(request.form['facebook_posts_last_6m']),
            # 'youtube_avg_views': float(request.form['youtube_avg_views']),
            'youtube_monetized': 1 if "youtube_monetization" in request.form else 0,
            'loan_amount': float(request.form['loan_amount']),
            'loan_duration': int(request.form['loan_duration'])
        }

        # Encode categorical levels to numeric
        # level_map = {'low': 0, 'medium': 1, 'high': 2}
        # form_data['linkedin_activity_level'] = level_map.get(form_data['linkedin_activity_level'].lower(), 0)
        # form_data['twitter_activity_level'] = level_map.get(form_data['twitter_activity_level'].lower(), 0)

        # Construct input for model
        input_data = pd.DataFrame([{
            'annual_income': form_data['annual_income'],
            'employment_length_years': form_data['employment_length_years'],
            'debt_to_income_ratio': form_data['debt_to_income_ratio'],
            'credit_score': form_data['credit_score'],
            'num_accounts': form_data['num_accounts'],
            'num_inquiries_6m': form_data['num_inquiries_6m'],
            'bankruptcy_flag': form_data['bankruptcy_flag'],
            'foreclosure_flag': form_data['foreclosure_flag'],
            'employment_status': form_data['employment_status'],
            'gender': form_data['gender'],
            'housing_status': form_data['housing_status'],
            'linkedin_activity_level': form_data['linkedin_activity_level'],
            'twitter_activity_level': form_data['twitter_activity_level'],
            'facebook_posts_last_6m': form_data['facebook_posts_last_6m'],
            'youtube_avg_views': form_data['youtube_avg_views'],
            'youtube_monetized': form_data['youtube_monetized']
        }])

        # Apply encoder and scaler
        input_encoded = woe_encoder.transform(input_data)
        input_scaled = scaler.transform(input_encoded)

        # Predictions
        predicted_loan_amount_lr = lr_loan.predict(input_scaled)[0]
        predicted_interest_rate_lr = lr_int.predict(input_scaled)[0]
        loan_approval_lr = lr_approval.predict(input_scaled)[0]

        predicted_loan_amount_lgbm = lgbm_loan_amount.predict(input_encoded)[0]
        predicted_interest_rate_lgbm = lgbm_interest_rate.predict(input_encoded)[0]
        loan_approval_lgbm = lgbm_model.predict(input_encoded)[0]

        # Save to customer and loan application datasets
        customer_df = pd.DataFrame([{
            "customer_id": customer_id,
            "first_name": form_data['first_name'],
            "last_name": form_data['last_name'],
            "email": form_data['email'],
            "gender": form_data['gender'],
            "employment_status": form_data['employment_status'],
            "employment_length_years": form_data['employment_length_years'],
            "housing_status": form_data['housing_status'],
            "annual_income": form_data['annual_income']
        }])

        loan_df = pd.DataFrame([{
            "application_id": application_id,
            "customer_id": customer_id,
            "loan_amount": form_data['loan_amount'],
            "loan_term_months": form_data['loan_duration'],
            "status": "Approved" if loan_approval_lgbm else "Rejected",
            "interest_rate": predicted_interest_rate_lgbm
        }])

        # customer_df.to_csv('database/customer_table.csv', mode='a', index=False, header=not os.path.exists('database/customer_table.csv'))
        # loan_df.to_csv('database/loan_application_table.csv', mode='a', index=False, header=not os.path.exists('database/loan_application_table.csv'))

        return render_template('results.html', 
                               linear_result=round(predicted_loan_amount_lr, 2), 
                               interest_rate=round(predicted_interest_rate_lr, 2), 
                               approval_status_lr="Approved" if loan_approval_lr else "Rejected",
                               approval_status_lgbm="Approved" if loan_approval_lgbm else "Rejected",
                               loan_amount_lgbm=round(predicted_loan_amount_lgbm, 2),
                               interest_rate_lgbm=round(predicted_interest_rate_lgbm, 2))

    return render_template('loan_form.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
