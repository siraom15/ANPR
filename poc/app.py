from flask import Flask, render_template, Response, redirect, url_for
import cv2
import time
import os
import mysql.connector

app = Flask(__name__)

# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="rootroot",
    database="car_detection_db"
)
cursor = db.cursor()

# Create our car classifier
car_classifier = cv2.CascadeClassifier('cars.xml')

# Create directory to save car images if it doesn't exist
output_dir = 'static/car-images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def detect_cars():
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    frame_count = 0

    while True:
        time.sleep(0.05)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray, 1.4, 2)

        if len(cars) > 0:
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            # Save to database
            cursor.execute("INSERT INTO car_images (path) VALUES (%s)", (frame_path,))
            db.commit()
            frame_count += 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_cars(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/car_images')
def car_images():
    cursor.execute("SELECT path, timestamp FROM car_images ORDER BY timestamp DESC")
    images = cursor.fetchall()
    return render_template('car_images.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
