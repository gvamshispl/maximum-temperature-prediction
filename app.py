from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the saved model and encoder
# Ensure these files are in the same folder as app.py
model = joblib.load("model.sav")
encoder = joblib.load("encoder.sav")

@app.route('/')
def index():
    # Now encoder.classes_ will exist because we saved a fitted encoder
    weather_options = encoder.classes_
    return render_template('index.html', weather_options=weather_options)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get data from the HTML form
            temp_min = float(request.form['temp_min'])
            wind = float(request.form['wind'])
            weather_text = request.form['weather']

            # 2. Use the encoder to transform the text to the correct number
            # We use .transform() here, NOT .fit_transform()
            weather_encoded = encoder.transform([weather_text])[0]

            # 3. Create DataFrame for the model
            input_df = pd.DataFrame([[temp_min, wind, weather_encoded]], 
                                    columns=['temp_min', 'wind', 'weather_encoded'])
            
            # 4. Make prediction
            prediction = model.predict(input_df)[0]
            
            return render_template('index.html', 
                                   prediction=round(prediction, 2), 
                                   weather_options=encoder.classes_)
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)