from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model and scaler
def load_model():
    if not os.path.exists('model.pkl'):
        raise FileNotFoundError("Model not found. Please run train_model.py first.")
    
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['scaler'], model_data['feature_names']

try:
    model, scaler, feature_names = load_model()
    print("✅ Model loaded successfully")
    print(f"📋 Features: {feature_names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, scaler, feature_names = None, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train_model.py first.'}), 500
    
    try:
        # Get form data
        holiday = int(request.form.get('holiday', 7))
        temp = float(request.form.get('temp', 15))
        rain = int(request.form.get('rain', 0))
        snow = int(request.form.get('snow', 0))
        
        # Get current date and time for additional features
        from datetime import datetime
        now = datetime.now()
        
        # Create feature vector in the same order as training
        features = [
            holiday,      # holiday
            temp,         # temp
            rain,         # rain
            snow,         # snow
            0,            # clouds (default to 0)
            1,            # clear (default to 1)
            0,            # mist (default to 0)
            now.day,      # day
            now.month,    # month
            now.year,     # year
            now.hour,     # hours
            now.minute,   # minutes
            now.second    # seconds
        ]
        
        # Ensure we have the right number of features
        if len(features) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features, got {len(features)}'}), 400
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'features_used': dict(zip(feature_names, features))
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features': feature_names if feature_names else []
    })

if __name__ == "__main__":
    if model is None:
        print("⚠️  Warning: Model not loaded. Please run train_model.py first.")
    app.run(debug=True, host='0.0.0.0', port=5000)
