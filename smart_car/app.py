# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np
from picamera2 import Picamera2

app = Flask(__name__)

# Initialize Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- GREEN LIGHT DETECTING ALGORITHM ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- BLUE LIGHT DETECTING ALGORITHM ---
        mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
        contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "GREEN LIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- RED LIGHT DETECTING ALGORITHM ---
        mask_red = cv2.add(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
                           cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255])))
        contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "RED LIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Packing frame into JPEG to streaming (MJPEG Stream)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield returned data to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Homepage link
@app.route('/')
def index():
    return render_template('index.html')

# Video link
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host='0.0.0.0' enable other devices in network to connect
    app.run(host='0.0.0.0', port=5000, debug=False)
