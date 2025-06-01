# app.py
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import joblib
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)

# Load model and setup MediaPipe
model_path = "hand_digit_classifier_sorted1.pkl"
model = joblib.load(model_path)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign-language')
def goto():
    return render_template('translation.html')

@app.route('/learn-sign-language')
def learn():
    return render_template('learn.html')

@app.route('/learn_0-9')
def learn0_9():
    return  render_template('learn0_9.html')

@app.route('/learn-bangla-alphabet')
def learnbanglaAlphabhet():
    return  render_template('learn-bangla-alphabet.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        frame = np.array(image)

        # Flip to simulate mirror and convert for MediaPipe
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                X = np.array(coords).reshape(1, -1)
                pred = int(model.predict(X)[0])
                return jsonify({'prediction': pred})
        else:
            # print("[INFO] No hand landmarks detected")
            return jsonify({'prediction': 'No hand detected'})
    except Exception as e:
        # print(f"[ERROR] {e}")
        return jsonify({'prediction': 'Error occurred'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
