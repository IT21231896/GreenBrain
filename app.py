from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import firebase_admin
import numpy as np
 
from firebase_admin import credentials, db as firebase_db

from datetime import datetime, timedelta
import os
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'IOT_AIApp001'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plantation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize Firebase
cred = credentials.Certificate("watering-7b4c7-firebase-adminsdk-ntnzb-ee048fb927.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://watering-7b4c7-default-rtdb.firebaseio.com/'
})

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    last_fertilizer_date = db.Column(db.String(20))

# Load all models
irrigation_model = joblib.load("models/irrigation_model_best.pkl")
irrigation_scaler = joblib.load("models/scaler_best.pkl")
irrigation_poly = joblib.load("models/poly_best.pkl")

fertilizer_rf_model = joblib.load("models/rf_model.pkl")
fertilizer_gb_model = joblib.load("models/gb_model.pkl")
fertilizer_scaler = joblib.load("models/scaler2.pkl")

harvest_model = joblib.load("models/harvest_model.pkl")
harvest_scaler = joblib.load("models/harvest_scaler.pkl")
min_growth_stage, max_growth_stage = joblib.load("models/growth_stage_range.pkl")
harvest_feature_names = joblib.load("models/feature_names.pkl")

 

def get_sensor_data():
    """Fetch sensor data from Firebase"""
    ref = firebase_db.reference('/sensors')
    data = ref.get()
    return data or {}



def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Prediction functions
def predict_irrigation(temperature, humidity, soil_moisture, light_level):
    input_data = pd.DataFrame([[temperature, humidity, soil_moisture, light_level]],
                            columns=['Temperature (°C)', 'Humidity (%)', 'Soil Moisture (%)', 'Light Level (lux)'])
    input_data_scaled = irrigation_scaler.transform(input_data)
    input_data_poly = irrigation_poly.transform(input_data_scaled)
    return irrigation_model.predict(input_data_poly)[0]

def predict_fertilizer(input_data):
    input_df = pd.DataFrame(input_data)
    input_df = pd.get_dummies(input_df, columns=['Fertilizer Type', 'Soil type', 'Growth stage'], drop_first=True)
    input_df = input_df.reindex(columns=fertilizer_scaler.feature_names_in_, fill_value=0)
    X_scaled = fertilizer_scaler.transform(input_df)
    quantity = fertilizer_rf_model.predict(X_scaled)[0]
    timing = int(round(fertilizer_gb_model.predict(X_scaled)[0]))
    return quantity, timing

def predict_harvest(input_data):
    input_df = pd.DataFrame(input_data, columns=harvest_feature_names)
    input_scaled = harvest_scaler.transform(input_df)
    predicted_days = harvest_model.predict(input_scaled)[0]
    return np.clip(predicted_days, 55, 90)

# Routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
        else:
            new_user = User(
                username=username,
                email=email,
                password=generate_password_hash(password, method='pbkdf2:sha256')

            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('auth/login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    sensor_data = get_sensor_data()
    return render_template('dashboard.html', sensor_data=sensor_data)

@app.route('/irrigation', methods=['GET', 'POST'])
@login_required
def irrigation():
    sensor_data = get_sensor_data()
    prediction = None
    
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            soil_moisture = float(request.form['soil_moisture'])
            light_level = float(request.form['light_level'])
            
            prediction = predict_irrigation(temperature, humidity, soil_moisture, light_level)
            prediction = f"{prediction:.2f} Liters"
        except ValueError:
            prediction = "Invalid input"
    
    return render_template('irrigation.html', 
                         prediction=prediction,
                         sensor_data=sensor_data)

@app.route('/fertilizer', methods=['GET', 'POST'])
@login_required
def fertilizer():
    user = User.query.get(session['user_id'])
    sensor_data = get_sensor_data()
    prediction = None
    
    if request.method == 'POST':
        if 'submit_fertilizer_form' in request.form:
            try:
                input_data = {
                    'Humidity (%)': [float(request.form['humidity'])],
                    'Temperature (°C)': [float(request.form['temperature'])],
                    'Area planted (ha)': [float(request.form['area_planted'])],
                    'Fertilizer Type': [request.form['fertilizer_type']],
                    'Soil type': [request.form['soil_type']],
                    'Growth stage': [request.form['growth_stage']]
                }
                
                quantity, weeks = predict_fertilizer(input_data)
                next_date = None
                
                if user.last_fertilizer_date:
                    date_obj = datetime.strptime(user.last_fertilizer_date, '%Y-%m-%d')
                    next_date = (date_obj + timedelta(weeks=weeks)).strftime('%Y-%m-%d')
                
                prediction = {
                    'quantity': round(quantity, 2),
                    'timing': weeks,
                    'next_date': next_date
                }
            except Exception as e:
                prediction = {'error': str(e)}
        
        elif 'submit_date_form' in request.form:
            user.last_fertilizer_date = request.form['last_fertilizer_date']
            db.session.commit()
            flash('Fertilizer date updated!', 'success')
    
    return render_template('fertilizer.html',
                         prediction=prediction,
                         last_date=user.last_fertilizer_date,
                         sensor_data=sensor_data)

@app.route('/harvest', methods=['GET', 'POST'])
@login_required
def harvest():
    sensor_data = get_sensor_data()
    prediction = None
    errors = []
    
    if request.method == 'POST':
        try:
            form_data = {
                'planting_date': request.form['planting_date'],
                'growth_stage': request.form['growth_stage'],
                'temperature': request.form['temperature'],
                'humidity': request.form['humidity'],
                'light_exposure': request.form['light_exposure'],
                'soil_moisture': request.form['soil_moisture'],
                'pesticide_used': request.form['pesticide_used']
            }
            
            # Validate inputs
            try:
                growth_stage = int(form_data['growth_stage'])
                if not (min_growth_stage <= growth_stage <= max_growth_stage):
                    errors.append(f"Growth Stage must be between {min_growth_stage} and {max_growth_stage} days")
            except:
                errors.append("Invalid Growth Stage value")
            
            try:
                temperature = float(form_data['temperature'])
                if not (10 <= temperature <= 40):
                    errors.append("Temperature must be between 10°C and 40°C")
            except:
                errors.append("Invalid Temperature value")
            
            if not errors:
                planting_date_ordinal = datetime.strptime(
                    form_data['planting_date'], "%Y-%m-%d").toordinal()
                
                new_data = pd.DataFrame([[
                    planting_date_ordinal,
                    int(form_data['growth_stage']),
                    float(form_data['temperature']),
                    float(form_data['humidity']),
                    float(form_data['light_exposure']),
                    float(form_data['soil_moisture']),
                    int(form_data['pesticide_used'])
                ]], columns=harvest_feature_names)
                
                predicted_days = predict_harvest(new_data)
                prediction = f"Predicted Harvest Days: {predicted_days:.0f} days"
        
        except Exception as e:
            errors.append(f"System error: {str(e)}")
    
    return render_template('harvest.html',
                         prediction=prediction,
                         errors=errors,
                         sensor_data=sensor_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)