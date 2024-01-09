from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask import flash
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model (replace 'your_model.joblib' with the actual filename)
XG_boost_pipeline = load('model/XG_boost_pipeline.joblib')
loaded_label_encoder = load('model/label_encoder_y.joblib')

# Initial empty DataFrame
df = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    global df

    # Get form data
    form_data = request.form.to_dict()

    # Convert form data to DataFrame
    temp_df = pd.DataFrame([form_data])

    # Combine the temporary DataFrame with the main DataFrame
    df = pd.concat([df, temp_df], ignore_index=True)

    # Infer data types for columns dynamically
    for column in df.columns:
        if df[column].dtype == 'O':
            df[column] = pd.to_numeric(df[column], errors='ignore')

    # Extract the last record for prediction
    last_record = df.iloc[-1].to_frame().T

    # Make prediction using the trained model (replace this with your actual prediction code)
    prediction = XG_boost_pipeline.predict(last_record)
    y_pred_original = loaded_label_encoder.inverse_transform(prediction)

    # Display the prediction as a popup message
    flash(f"Prediction: {y_pred_original}")
    print(y_pred_original[0])

    return render_template('submit.html', prediction=y_pred_original[0])


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'  # Needed for flash messages
    app.run(debug=True)
