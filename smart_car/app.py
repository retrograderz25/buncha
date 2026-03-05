# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np
import subprocess

app = Flask(__name__)

def generate_frames():
    width = 640
    height = 480
    command = [
        'rpicam-vid',
        '-t', '0', '-n',
        '--width', str(width),
        '--height', str(height),
        '--framerate', '30',
        '--codec', 'yuv420',
        '-o', '-'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_size = int(width * height * 1.5)

    try:
        while True:
            raw_data = process.stdout.read(frame_size)
            if len(raw_data) != frame_size:
                print("error")
                continue

            yuv_frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(height * 1.5), width))

            frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # --- GREEN LIGHT DETECTING ---
            mask_green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
            contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, "GREEN LIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # --- RED LIGHT DETECTING ---
            mask_red = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            )
            contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, "RED LIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        process.terminate()
        process.wait()

# Homepage link
@app.route('/')
def index():
    return render_template('index.html')

# Video link
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
