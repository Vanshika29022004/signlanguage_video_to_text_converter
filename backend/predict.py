# libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thankyou', 'loveyou', ' '
])

model = load_model('action.h5')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = np.array(request.json['sequence'])  # Expecting shape (30, 1662)
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    predicted_action = actions[np.argmax(res)]
    return jsonify({'prediction': predicted_action})

if __name__ == '__main__':
    app.run(port=5000)
