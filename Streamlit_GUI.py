import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\Dell\Downloads\pollen_luxembourg_Dataset.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data

data = load_data()

# Prepare features and target variable
X = data[['MaxAirTempC', 'MinAirTempC', 'PrecipitationC', 'Month', 'DayOfYear']]
pollen_columns = ['Ambrosia', 'Artemisia', 'Asteraceae', 'Alnus', 'Betula', 'Ericaceae', 
                  'Carpinus', 'Castanea', 'Quercus', 'Chenopodium', 'Cupressaceae', 
                  'Acer', 'Fraxinus', 'Gramineae', 'Fagus', 'Juncaceae', 'Aesculus', 
                  'Larix', 'Corylus', 'Juglans', 'Umbellifereae', 'Ulmus', 'Urtica', 
                  'Rumex', 'Populus', 'Pinaceae', 'Plantago', 'Platanus', 'Salix', 
                  'Cyperaceae', 'Filipendula', 'Sambucus', 'Tilia']
y = data[pollen_columns].sum(axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Calculate threshold for the alert system (90th percentile)
threshold = np.percentile(y_train, 90)

# Prediction function
def predict_pollen_concentration(weather_data):
    prediction = rf_model.predict([weather_data])
    if prediction > threshold:
        alert = "High Pollen Alert! Predicted Pollen Concentration exceeds the threshold."
    else:
        alert = "Pollen levels are within normal range."
    return prediction[0], alert

# Streamlit UI layout
st.title('Pollen Concentration Prediction and Alert System')

st.write("""
### Enter Weather Data for Pollen Prediction:
""")

MaxAirTempC = st.number_input("Max Air Temperature (°C):", min_value=-30.0, max_value=50.0, value=25.0)
MinAirTempC = st.number_input("Min Air Temperature (°C):", min_value=-30.0, max_value=50.0, value=15.0)
PrecipitationC = st.number_input("Precipitation (mm):", min_value=0.0, max_value=500.0, value=2.0)
Month = st.number_input("Month (1-12):", min_value=1, max_value=12, value=6)
DayOfYear = st.number_input("Day of Year (1-365):", min_value=1, max_value=365, value=170)

if st.button('Predict Pollen Concentration'):
    # Collect input data and make a prediction
    new_weather_data = [MaxAirTempC, MinAirTempC, PrecipitationC, Month, DayOfYear]
    predicted_concentration, alert_message = predict_pollen_concentration(new_weather_data)

    st.write(f"**Predicted Pollen Concentration:** {predicted_concentration}")
    st.write(f"**Alert:** {alert_message}")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(['Predicted Concentration'], [predicted_concentration], 
           color='red' if predicted_concentration > threshold else 'green')
    ax.axhline(y=threshold, color='black', linestyle='--', label='Alert Threshold')
    ax.set_ylabel('Pollen Concentration')
    ax.set_title('Pollen Concentration Prediction')
    ax.legend()
    st.pyplot(fig)
