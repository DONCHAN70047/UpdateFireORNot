# Smoke - 200-300
# Temp - 20 - 40

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import os
import serial
import time
import psycopg2
import threading

# Load AI Model
model = tf.keras.models.load_model('model/model.h5')

# Flask App Initialization
app = Flask(__name__)
UPLOAD_FOLDER = "captured_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serial Connection for Arduino
try:
    ser = serial.Serial('COM5', 9600, timeout=1)
    ser.flush()
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    ser = None  # Prevent errors if serial fails

# Database Credentials
DB_CONFIG = {
    "host": "localhost",
    "dbname": "HotelSafety",
    "user": "postgres",
    "password": "Arpan@123",
    "port": "5432"
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print("Database connection error:", e)
        return None

def insert_sensor_data(temp, smokeTemp):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        query = "INSERT INTO firedetailsdetection (camera1, camera2) VALUES (%s, %s);"
        cursor.execute(query, (temp, smokeTemp))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error inserting data:", e)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data["image"].split(",")[-1]
    image_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
    with open(image_path, "wb") as image_file:
        image_file.write(base64.b64decode(image_data))

    camera_confidence = predict_from_image(model, image_path)
    FireArray = fetch_sensor_data()
    #delete_oldest_record()
    tempa, smoke = (FireArray[-1, 1], FireArray[-1, 2]) if FireArray.size > 0 else (25, 50)

    fire_risk, smoke_percentage, temp_percentage = fire_risk_percentageF(camera_confidence, smoke, tempa)
    print(f"🔥 Fire Risk: {fire_risk}% | 💨 Smoke Level: {smoke_percentage}% | 🌡 Temperature: {temp_percentage}%")
    
    #Logic
    resultPercent = fire_risk_percentage(camera_confidence, smoke, tempa)
        
    return jsonify({'fire_risk_percentage': resultPercent})

def fetch_sensor_data():
    conn = get_db_connection()
    if conn is None:
        return np.array([])
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, camera1, camera2 FROM firedetailsdetection ORDER BY id;")
        target = cursor.fetchall()
        cursor.close()
        conn.close()
        return np.array(target, dtype=float) if target else np.array([])
    except Exception as e:
        print("Error fetching data:", e)
        return np.array([])
def delete_oldest_record():
    """ Delete the oldest record from the database to keep data manageable. """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM firedetailsdetection WHERE id = (SELECT MIN(id) FROM firedetailsdetection);")
        conn.commit()
        cursor.close()
        conn.close()
        print(">>Oldest record deleted successfully.")
    except Exception as e:
        print("Error deleting record:", e)


def fire_risk_percentageF(camera_confidence, smoke_level, temperature):
    fire_percentage = round((1 - camera_confidence) * 100, 2)
    smoke_percentage = round((smoke_level / 500) * 100, 2)
    temp_percentage = round((temperature / 100) * 100, 2)
    return fire_percentage, smoke_percentage, temp_percentage

def fire_risk_percentage(camera_confidence, smoke_level, temperature):
    fire_percentage = (1 - camera_confidence) * 100  

    max_smoke_level = 300  
    smoke_percentage = max(0, min(100, ((smoke_level - 200) / (max_smoke_level - 200)) * 100)) 

    min_temp = 20  
    max_temp = 26
    temp_percentage = max(0, min(100, ((temperature - min_temp) / (max_temp - min_temp)) * 100)) 
    total_risk = min(100, (fire_percentage + smoke_percentage + temp_percentage) / 3)

    print(f"🔥 Fire Prediction: {fire_percentage:.2f}%")
    print(f"💨 Smoke Level: {smoke_percentage:.2f}%")
    print(f"🌡️ Temperature: {temp_percentage:.2f}%")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>🚨 Total Fire Risk: {total_risk:.2f}%")

    return round(total_risk, 2)


def predict_from_image(model, image_path="captured_images/captured_image.png"):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        fire_percentage = round((1 - prediction) * 100, 2)
        print(f"🔥 Fire Percentage: {fire_percentage}%")
        return prediction
    except Exception as e:
        print(f"Error in image prediction: {e}")
        return None

def sensor_data_collection():
    while True:
        if ser and ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    temperatureC, analogTemp = map(float, line.split(','))
                    insert_sensor_data(temperatureC, analogTemp)
                    delete_oldest_record()
                except ValueError:
                    continue
        time.sleep(1)

sensor_thread = threading.Thread(target=sensor_data_collection, daemon=True)
sensor_thread.start()

if __name__ == '__main__':
    app.run(debug=True)
