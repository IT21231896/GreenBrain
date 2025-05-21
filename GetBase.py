import firebase_admin
from firebase_admin import credentials, db
import time

# Initialize Firebase
cred = credentials.Certificate("watering-7b4c7-firebase-adminsdk-ntnzb-ee048fb927.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://watering-7b4c7-default-rtdb.firebaseio.com/'
})

# Realtime Database path
sensor_ref = db.reference('/sensors')  # This should point to the '/sensors' path

def print_real_time_data():
    while True:
        try:
            # Retrieve real-time sensor data
            sensor_data = sensor_ref.get()

            if sensor_data:
                print("Sensor Data:")
                print(f"  Distance: {sensor_data.get('distance', 'N/A')}")
                print(f"  Humidity: {sensor_data.get('humidity', 'N/A')}")
                print(f"  Light Level: {sensor_data.get('lightLevel', 'N/A')}")
                print(f"  Soil Moisture: {sensor_data.get('soilMoisture', 'N/A')}")
                print(f"  Temperature: {sensor_data.get('temperature', 'N/A')}")
            else:
                print("No sensor data available")

            print("\n--- Waiting for next update ---\n")
            time.sleep(1)

        except KeyboardInterrupt:
            print("Program terminated.")
            break

if __name__ == "__main__":
    print_real_time_data()
