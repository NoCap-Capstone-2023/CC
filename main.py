from flask import Flask, request, jsonify
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load your trained model
model = load_model('model_hand.h5')  # Replace 'model_lab.h5' with your actual model file path

def get_letters(img_path):
    letters = []
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y:y + h, x:x + w]
            
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 28, 28, 1)

            # Predict using the loaded model
            y_pred = model.predict(thresh)

            # Extract the predicted character based on the maximum probability
            predicted_char_index = np.argmax(y_pred)
            predicted_char = chr(65 + predicted_char_index)  # Assuming uppercase English letters; adjust if needed
            letters.append((predicted_char, (x, y, w, h)))

    letters = sorted(letters, key=lambda x: x[1][0])

    combined_word = ''
    for letter, _ in letters:
        combined_word += letter

    return combined_word

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save('uploaded_image.jpg')  # Save the uploaded image
            result = get_letters('uploaded_image.jpg')  # Process the uploaded image
            return jsonify({'result': result})
    return jsonify({'error': 'No file uploaded'})

@app.route('/')
def hello():
        return 'Hello, World'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)