import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
engine = pyttsx3.init()

speech_queue = queue.Queue()  # Queue for speech messages

# Global variables for squat counting
counter = 0
stage = None

# Function to process squat detection
def detect_squat(image):
    global counter, stage

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get necessary landmarks
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y

        # Calculate knee angle
        angle = np.arctan2(knee - hip, ankle - knee) * 180 / np.pi

        # Squat detection logic
        if angle > 160:  # Standing position
            stage = "up"
        elif angle < 90 and stage == "up":  # Squat position
            stage = "down"
            counter += 1
            speech_queue.put(f"Squat {counter}")  # Add speech message to queue

    return image

# Function to generate video feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_squat(frame)  # Process frame for squat detection

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset_counter():
    global counter
    counter = 0
    return jsonify({"message": "Counter reset!"})

# Speech worker to process queued messages
def speech_worker():
    while True:
        text = speech_queue.get()
        if text:
            engine.say(text)
            engine.startLoop(False)  # Start loop without blocking
            while engine._inLoop:
                engine.iterate()
        speech_queue.task_done()

# Start speech synthesis in a separate thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

if __name__ == '__main__':
    app.run(debug=True)
