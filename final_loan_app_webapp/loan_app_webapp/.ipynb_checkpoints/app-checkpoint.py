# Flask app logic goes here
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os
import webbrowser

app = Flask(__name__)
app.secret_key = 'batman_secret_key'

# Load models
lr_model = joblib.load('models/linear_regression_model.pkl')
lgbm_model = joblib.load('models/lgbm_model.pkl')

# Routes
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    credentials = pd.read_csv('database/credentials.csv')
    user = credentials[(credentials['username'] == username) & (credentials['password'] == password)]

    if not user.empty:
        session['username'] = username
        # return render_template('loan_form.html')
        return redirect(url_for('loan_form'))
    else:
        return render_template('register.html')
        # return "Invalid credentials. Please try again."

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
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        dob = request.form['dob']
        email = request.form["email"]
        phone = request.form["phone"]
        address = request.form["address"]
        profession = request.form["profession"]
        company = request.form["company"]
        income = request.form["income"]
        loan_amount = request.form["loan_amount"]
        loan_duration = request.form["loan_duration"]
    
        
        # Social Media Sharing
        share_twitter = 'Y' if "share_twitter" in request.form else 'N'
        twitter_url = request.form["twitter_url"] or ''
    
        share_facebook = 'Y' if "share_facebook" in request.form else 'N'    
        facebook_url = request.form["facebook_url"] or '' 
    
        share_linkedin = 'Y' if "share_linkedin" in request.form else 'N'        
        linkedin_url = request.form["linkedin_url"] or ''
    
        share_youtube = 'Y' if "share_youtube" in request.form else 'N'        
        youtube_url = request.form["youtube_url"] or ''
        youtube_monetization = 'Y' if "youtube_monetization" in request.form else 'N'


        # Save to customer_table
        customer_df = pd.DataFrame([{
            "name": first_name + " " + last_name ,
            "email": email,
            "profession": profession,
            "income": income,
            "share_linkedin": share_linkedin
        }])
        

        # Save to loan_application_table
        loan_df = pd.DataFrame([{
            "name": first_name + " " + last_name,
            "income": income,
            "loan_approved": 0  # dummy placeholder
        }])
        

        # profession_map = ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Other']
        # profession = profession if profession in profession_map else 'Other'
        # prof_features = {f'profession_{p}': 1 if profession == p else 0 for p in profession_map}
        # input_data = pd.DataFrame([{
        #     "income": income,
        #     "share_linkedin": share_linkedin,
        # }])

        # # Make sure all columns exist
        # for col in ['profession_Engineer', 'profession_Teacher', 'profession_Doctor', 'profession_Artist']:
        #     if col not in input_data:
        #         input_data[col] = 0



        # Prediction
        # profession_map = ['Engineer', 'Teacher', 'Doctor', 'Artist']
        prof_features = 0 if profession == "" else 1
        linkedin_features = 1 if share_linkedin == "Y" else 0 
        data = pd.DataFrame([{
            "name" : first_name + " " + last_name,
            "email": email,
            "profession" : prof_features,
            "income": int(income),
            "share_linkedin": linkedin_features
            
        }])

        input_data = data.drop(columns=['name', 'email'])
        
        print(input_data) 
        # pd.get_dummies(customer_df, columns=['profession'])
        # # Make sure all columns exist
        # for col in ['profession_Engineer', 'profession_Teacher', 'profession_Doctor', 'profession_Artist']:
        #     if col not in input_data:
        #         input_data[col] = 0

        lr_result = lr_model.predict(input_data)[0]
        lgbm_result = lgbm_model.predict(input_data)[0]

        customer_df.to_csv('database/customer_table.csv', mode='a', index=False, header=False)
        loan_df.to_csv('database/loan_application_table.csv', mode='a', index=False, header=False)
        return render_template('results.html', linear_result=lr_result, lgbm_result=lgbm_result)
        # return render_template("results.html",
        #                        lr_result="Approved" if lr_result >= 0.5 else "Rejected",
        #                        lgbm_result="Approved" if lgbm_result == 1 else "Rejected")

    return render_template('loan_form.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
