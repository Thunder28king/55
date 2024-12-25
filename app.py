from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime

# Load the trained model
MODEL_PATH = 'retrained_model_with_timestamps.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Feature Engineering Functions
def calculate_entropy(text):
    probabilities = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * pd.np.log2(p) for p in probabilities if p > 0)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    gid = data.get('GID')
    bet_amount = data.get('BetAmount')
    
    if not gid or not bet_amount:
        return jsonify({'error': 'Both GID and BetAmount are required'}), 400
    
    # Calculate features
    gid_entropy = calculate_entropy(gid)
    gid_length = len(gid)
    current_time = datetime.utcnow()
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    time_weight = (hour * 0.5) + (minute * 0.3) + (second * 0.2)
    
    # Prepare input data
    input_features = pd.DataFrame({
        'BetAmount': [bet_amount],
        'GID_Entropy': [gid_entropy],
        'GID_Length': [gid_length],
        'Day': [day],
        'Hour': [hour],
        'Minute': [minute],
        'Second': [second],
        'TimeWeight': [time_weight]
    })
    
    # Make prediction
    predicted_class = model.predict(input_features)[0]
    result = predicted_class
    
    return jsonify({
        'GID': gid,
        'BetAmount': bet_amount,
        'Day': day,
        'Hour': hour,
        'Minute': minute,
        'Second': second,
        'TimeWeight': time_weight,
        'PrimaryResult': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
