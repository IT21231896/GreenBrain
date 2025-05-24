from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import firebase_admin
import numpy as np
import math
from flask import jsonify

from firebase_admin import credentials, db as firebase_db

from datetime import datetime, timedelta

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
model = joblib.load("models/irrigation_model_best.pkl")
scaler = joblib.load("models/scaler_best.pkl")
poly = joblib.load("models/poly_best.pkl")


fertilizer_rf_model = joblib.load("models/rf_model.pkl")
fertilizer_gb_model = joblib.load("models/gb_model.pkl")
fertilizer_scaler = joblib.load("models/scaler2.pkl")

harvest_model = joblib.load("models/harvest_model.pkl")
harvest_scaler = joblib.load("models/harvest_scaler.pkl")
min_growth_stage, max_growth_stage = joblib.load("models/growth_stage_range.pkl")
harvest_feature_names = joblib.load("models/feature_names.pkl")

sensor_ref = firebase_db.reference('/sensors')

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


# Prediction Logic
def predict_watering(temperature, humidity, soil_moisture, light_level):
    input_data = pd.DataFrame([[temperature, humidity, soil_moisture, light_level]],
                              columns=['Temperature (°C)', 'Humidity (%)', 'Soil Moisture (%)', 'Light Level (lux)'])
    input_data_scaled = scaler.transform(input_data)
    input_data_poly = poly.transform(input_data_scaled)
    predicted_water_level = model.predict(input_data_poly)
    return predicted_water_level[0]

def get_prediction_reason(temperature, humidity, soil_moisture, light_level):
    if temperature > 35:
        return "Critical Warning: High temperature detected. Immediate action is required to prevent heat stress!"
    elif temperature > 30:
        return "Alert: Elevated temperature detected. Consider providing shade and increasing watering frequency."
    elif temperature < 15:
        return "Notice: Low temperature detected. Ensure protection against frost and consider using row covers."
    elif humidity < 40:
        return "Alert: Low humidity detected. Increase humidity around plants to prevent blossom drop."
    elif soil_moisture < 20:
        return "Warning: Low soil moisture detected. Increase watering to maintain optimal soil moisture levels."
    elif light_level < 6:
        return "Notice: Insufficient light detected. Ensure plants receive adequate sunlight for healthy growth."
    else:
        return "Conditions are optimal for irrigation. No immediate action required."


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



@app.route("/irrigation", methods=["GET", "POST"])
@login_required
def irrigation():
    predicted_water_level = None
    total_water_requirement = None
    prediction_reason = None
    sensor_data = get_sensor_data()
    count_method = 'direct'
    greenhouse_area = None
    plant_gap = 1.5

    temperature = humidity = soil_moisture = light_level = plant_count = None

    if sensor_data:
        temperature = sensor_data.get('temperature', '')
        humidity = sensor_data.get('humidity', '')
        soil_moisture = sensor_data.get('soilMoisture', '')
        light_level = sensor_data.get('lightLevel', '')

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            soil_moisture = float(request.form["soil_moisture"])
            light_level = float(request.form["light_level"])

            count_method = request.form.get("count_method", "direct")

            if count_method == "area":
                greenhouse_area = float(request.form["greenhouse_area"])
                plant_gap = float(request.form["plant_gap"])
                plant_count = math.floor(greenhouse_area / (plant_gap ** 2))
            else:
                plant_count = int(request.form["plant_count"])

            predicted_water_level = predict_watering(temperature, humidity, soil_moisture, light_level)
            total_water_requirement = predicted_water_level * plant_count
            prediction_reason = get_prediction_reason(temperature, humidity, soil_moisture, light_level)
        except ValueError:
            predicted_water_level = "Invalid input"
            total_water_requirement = "Invalid input"
            prediction_reason = "Invalid input"

    import datetime
    timestamp = datetime.datetime.now().isoformat()

    log_ref = firebase_db.reference('/prediction_logs')

    log_ref.push({
        "timestamp": timestamp,
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "light_level": light_level,
        "plant_count": plant_count,
        "per_plant_water": predicted_water_level,
        "total_water": total_water_requirement
    })

    return render_template("irrigation.html",
                           sensor_data=sensor_data,
                           predicted_water_level=predicted_water_level,
                           total_water_requirement=total_water_requirement,
                           prediction_reason=prediction_reason,
                           temperature=temperature,
                           humidity=humidity,
                           soil_moisture=soil_moisture,
                           light_level=light_level,
                           plant_count=plant_count,
                           count_method=count_method,
                           greenhouse_area=greenhouse_area,
                           plant_gap=plant_gap
                           )


@app.route('/sensor_data')
@login_required
def sensor_data():
    return render_template('SensorData.html')



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


@app.route('/history')
@login_required
def history():
    log_ref = firebase_db.reference('/prediction_logs')

    logs = log_ref.get()

    data = []
    if logs:
        for log in logs.values():
            # Defensive conversion
            try:
                log["per_plant_water"] = float(log.get("per_plant_water", 0))
                log["total_water"] = float(log.get("total_water", 0))
            except (ValueError, TypeError):
                log["per_plant_water"] = 0
                log["total_water"] = 0
            data.append(log)

    # Sort entries newest first
    data.sort(key=lambda x: x['timestamp'], reverse=True)

    return render_template('history.html', logs=data)


@app.route('/download_prediction_logs')
@login_required
def download_logs():
    ref = firebase_db.reference('/prediction_logs')

    logs = ref.get()

    if not logs:
        return "No logs found", 404

    import csv
    from io import StringIO
    si = StringIO()
    writer = csv.writer(si)

    # Write header
    writer.writerow(["timestamp", "temperature", "humidity", "soil_moisture", "light_level", "plant_count", "per_plant_water", "total_water"])

    # Write rows
    for log in logs.values():
        writer.writerow([
            log.get("timestamp"),
            log.get("temperature"),
            log.get("humidity"),
            log.get("soil_moisture"),
            log.get("light_level"),
            log.get("plant_count"),
            log.get("per_plant_water"),
            log.get("total_water")
        ])

    from flask import Response
    output = si.getvalue()
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=prediction_logs.csv"})

@app.route('/get_tank_height', methods=['GET'])
@login_required
def get_tank_height():
    ref = firebase_db.reference('/')
    tank_height = ref.child('tank_height_cm').get()

    if tank_height is None:
        tank_height = 100
    return jsonify({'tank_height_cm': tank_height})

@app.route('/set_tank_height', methods=['POST'])
@login_required
def set_tank_height():
    try:
        new_height = int(request.form['height'])
        if new_height <= 0:
            return jsonify({'message': 'Height must be positive'}), 400

        ref = firebase_db.reference('/')
        ref.update({'tank_height_cm': new_height})
        return jsonify({'message': 'Tank height updated successfully'})
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/update_irrigation_automation', methods=['POST'])
@login_required
def update_irrigation_automation():
    status = request.form.get('status')
    sensor_ref.update({'automation_irrigation': int(status)})
    return jsonify({"message": "Irrigation automation updated", "status": status})


@app.route('/update_alarm_automation', methods=['POST'])
@login_required
def update_alarm_automation():
    status = request.form.get('status')
    sensor_ref.update({'automated_alarm': int(status)})
    return jsonify({"message": "Alarm automation updated", "status": status})

@app.route('/update_fan_automation', methods=['POST'])
@login_required
def update_fan_automation():
    status = request.form.get('status')
    sensor_ref.update({'fan_automation': int(status)})
    return jsonify({"message": "Fan automation updated", "status": status})

@app.route('/update_buzzer_status', methods=['POST'])
@login_required
def update_buzzer_status():
    status = request.form.get('status')
    sensor_ref.update({'buzzer_status': int(status)})
    return jsonify({"message": "Buzzer status updated", "status": status})

@app.route('/update_vent_automation', methods=['POST'])
@login_required
def update_vent_automation():
    status = request.form.get('status')
    sensor_ref.update({'vent_status': int(status)})
    return jsonify({"message": "Ventilation automation updated", "status": status})

@app.route('/sensor_overview')
@login_required
def sensor_overview():
    return render_template('SensorData.html')


@app.route('/get_sensor_data_realtime', methods=['GET'])
@login_required
def get_sensor_data_realtime():
    sensor_data = sensor_ref.get()
    alerts = []

    if sensor_data:
        humidity = float(sensor_data.get('humidity', 0))
        temperature = float(sensor_data.get('temperature', 0))
        soil_moisture = float(sensor_data.get('soilMoisture', 0))
        water_level = float(sensor_data.get('waterLevel', 0))
        tank_height = firebase_db.reference('/tank_height_cm').get() or 100
        distance = float(sensor_data.get('distance', 0))
        tank_percent = ((tank_height - distance) / tank_height) * 100
        automation_irrigation = int(sensor_data.get('automation_irrigation', 0))
        fan_automation = int(sensor_data.get('fan_automation', 0))
        automated_alarm = int(sensor_data.get('automated_alarm', 0))


        #Alarm Buzzer Automation Conditions
        buzzer_triggered = False
        if automated_alarm == 1:
            if water_level > 1000:
                alerts.append("🚨 Water overflow detected! Buzzer activated.")
                buzzer_triggered = True
            if tank_percent < 10:
                alerts.append("⚠️ Water tank is below 10%. Buzzer activated.")
                buzzer_triggered = True
            if temperature > 40:
                alerts.append("🔥 High temperature! Buzzer activated.")
                buzzer_triggered = True
            if soil_moisture > 3500 and automation_irrigation == 0:
                alerts.append("🌱 Soil is dry. Enable auto irrigation or water manually. Buzzer activated.")
                buzzer_triggered = True
            else:
                alerts.append("⚠️ Buzzer remains OFF.Because All conditions are optimal")
        else:
            sensor_ref.update({'buzzer_status': 0})

        sensor_ref.update({'buzzer_status': 1 if buzzer_triggered else 0})



        # Irrigation Motor Automation Logic
        if automation_irrigation == 1:

            #if automation_irrigation On the firebase value is 1
            sensor_ref.update({'device1_status': 1})

            # Check water tank level is Optimal
            if tank_percent >= 20:
                # Start irrigation if soil is dry
                if soil_moisture > 2500:
                    alerts.append("Irrigation started: Soil moisture is high, and water is available.")
                else:
                    alerts.append("ℹ️ Soil moisture is within optimal range. No need for irrigation at this time.")
            else:
                alerts.append("🚫 Warning: Water tank level is below 20%. Please refill the tank soon.")

            # Alert if plant pot water level is high
            if water_level > 1000:
                alerts.append("⚠️ Warning: Water level in the plant pot is high. Monitor for potential overflow.")

        else:
            # Turn off irrigation if automation is disabled
            sensor_ref.update({'device1_status': 0})



        #Vemdilation Fan Automation Conditions
        if fan_automation == 1:

            sensor_ref.update({'device2_status': 1})

            if temperature > 35 or humidity < 70:
                alerts.append("🌬️ Fan turned ON: Temperature is above 35°C, cooling in progress.")
            elif temperature <= 35:
                alerts.append("✅ Fan turned OFF: Temperature is at or below 35°C, cooling not needed.")

        else:
            sensor_ref.update({'device2_status': 0})


        data = {
            'distance': sensor_data.get('distance', 'N/A'),
            'waterLevel': sensor_data.get('waterLevel', 'N/A'),
            'humidity': sensor_data.get('humidity', 'N/A'),
            'lightLevel': sensor_data.get('lightLevel', 'N/A'),
            'soilMoisture': sensor_data.get('soilMoisture', 'N/A'),
            'temperature': sensor_data.get('temperature', 'N/A'),
            'device1_status': sensor_data.get('device1_status', 'N/A'),
            'device2_status': sensor_data.get('device2_status', 'N/A'),
            'fan_automation': fan_automation,
            'automation_irrigation': automation_irrigation,
            'automated_alarm': automated_alarm,
            'vent_status': int(sensor_data.get('vent_status', 0)),
            'alerts': alerts,
            'buzzer_status': int(sensor_data.get('buzzer_status', 0))
        }

    else:
        data = {'message': 'No sensor data available'}
    return jsonify(data)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)