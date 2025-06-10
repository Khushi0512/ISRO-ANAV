from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import math
import scipy.ndimage
import time
import threading
import socket
import json
from pymavlink import mavutil

app = Flask(__name__)

# ---------- Camera Setup ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)})
config["buffer_count"] = 3
picam2.configure(config)
picam2.start()
time.sleep(2)

# ---------- Crack Detection Functions ----------
def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi / 4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# ---------- MJPEG Streaming ----------
def generate_frames():
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = picam2.capture_array()
        frame = frame[:, :, [2, 1, 0]]
        frame = frame.astype('uint8').copy()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image / 255.0

        with_nmsup = True
        fudgefactor = 1.3
        sigma = 21
        kernel = 2 * math.ceil(2 * sigma) + 1

        blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
        gray_image = cv2.subtract(gray_image, blur)

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        threshold = 4 * fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0

        if with_nmsup:
            mag = orientated_non_max_suppression(mag, ang)

        mag[mag > 0] = 255
        mag = mag.astype(np.uint8)

        contours, _ = cv2.findContours(mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FPS Overlay
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1.0:
            fps = frame_count
            frame_count = 0
            prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<h1>Live Crack Detection</h1><img src="/video_feed">'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- Joystick + Pixhawk Communication ----------
def joystick_to_pixhawk():
    # Connect to Pixhawk
    try:
        master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
        print("Waiting for Pixhawk heartbeat...")
        master.wait_heartbeat()
        print("✅ Pixhawk connected!")
    except Exception as e:
        print("❌ Failed to connect to Pixhawk:", e)
        return

    # Set up UDP listener
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"🎮 Listening for joystick data on UDP port {UDP_PORT}...")

    armed = False

    while True:
        data, addr = sock.recvfrom(1024)
        try:
            controller = json.loads(data.decode())
            print(f"\n🎮 From {addr}:")
            print("  Axes:", controller['axes'])
            print("  Buttons:", controller['buttons'])

            # Arm on button 0
            if controller['buttons'][0] == 1 and not armed:
                print("🟢 Arming Pixhawk...")
                master.arducopter_arm()
                master.motors_armed_wait()
                print("✅ Pixhawk armed!")
                armed = True

            # Disarm on button 1
            if controller['buttons'][1] == 1 and armed:
                print("🔴 Disarming Pixhawk...")
                master.arducopter_disarm()
                master.motors_disarmed_wait()
                print("✅ Pixhawk disarmed!")
                armed = False

        except Exception as e:
            print("⚠️ Error in joystick handler:", e)

# ---------- Threading ----------
if __name__ == '__main__':
    # Start joystick thread
    thread = threading.Thread(target=joystick_to_pixhawk)
    thread.daemon = True
    thread.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=8080)
