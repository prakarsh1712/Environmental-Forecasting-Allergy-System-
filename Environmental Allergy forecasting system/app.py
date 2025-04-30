import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from city_safety_model import CitySafetyModel, load_data, CitySafetyDataset
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model(sequence_length, n_features):
    try:
        if not os.path.exists('best_model.pth'):
            st.error("Error: No pretrained model found. Please train the model first using city_safety_model.py")
            return None
            
        if st.session_state.model is None:
            model = CitySafetyModel(sequence_length, n_features)
            model.load_state_dict(torch.load('best_model.pth'))
            model.eval()
            st.session_state.model = model
            
        return st.session_state.model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_city_data():
    try:
        city_data = pd.read_csv('dataset/Indian Cities Database.csv')
        return city_data
    except Exception as e:
        st.error(f"Error loading city data: {str(e)}")
        return None

def create_prediction_sequence(data, features, sequence_length, current_values):
    try:
        # Create a sequence with historical data and current values
        sequence = data[features].iloc[-sequence_length+1:].values
        current_row = np.array([current_values[feature] for feature in features])
        sequence = np.vstack([sequence, current_row])
        return sequence
    except Exception as e:
        st.error(f"Error creating prediction sequence: {str(e)}")
        return None

def make_prediction(model, sequence):
    try:
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            prediction = model(sequence_tensor)
            return prediction.item()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="City Safety Prediction System", layout="wide")
    
    # Title and description
    st.title("🌆 Environmental Allergy Forecasting System")
    st.markdown("""
    This system predicts whether a city is safe based on environmental factors and pollen levels.
    Select a city and enter the current conditions below to get a prediction.
    """)
    
    try:
        # Load city data
        city_data = load_city_data()
        if city_data is None:
            return
            
        # City selection
        st.subheader("Select City")
        selected_city = st.selectbox(
            "Choose a city",
            options=city_data['City'].unique(),
            index=0
        )
        
        # Get city details
        city_info = city_data[city_data['City'] == selected_city].iloc[0]
        
        # Display city information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latitude", f"{city_info['Lat']:.4f}°")
        with col2:
            st.metric("Longitude", f"{city_info['Long']:.4f}°")
        with col3:
            st.metric("State", city_info['State'])
        
        # Load data and model
        data, features = load_data()
        sequence_length = 10
        model = load_model(sequence_length, len(features))
        
        if model is None:
            st.error("Failed to load model. Please ensure you have trained the model first.")
            return
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Conditions")
            # Temperature inputs
            max_temp = st.slider("Maximum Temperature (°C)", -10.0, 45.0, 25.0)
            min_temp = st.slider("Minimum Temperature (°C)", -20.0, 35.0, 15.0)
            precipitation = st.slider("Precipitation (mm)", 0.0, 200.0, 0.0)
            
            # Location inputs (pre-filled with city coordinates)
            lat = st.number_input("Latitude", -90.0, 90.0, float(city_info['Lat']), disabled=True)
            long = st.number_input("Longitude", -180.0, 180.0, float(city_info['Long']), disabled=True)
        
        with col2:
            st.subheader("Pollen Levels")
            # Pollen inputs
            pollen_columns = ['Ambrosia', 'Artemisia', 'Asteraceae', 'Alnus', 'Betula', 
                             'Ericaceae', 'Carpinus', 'Castanea', 'Quercus', 'Chenopodium']
            
            pollen_values = {}
            for pollen in pollen_columns:
                pollen_values[pollen] = st.slider(f"{pollen} Pollen Count", 0, 100, 0)
        
        # Calculate derived features
        temp_range = max_temp - min_temp
        current_month = datetime.now().month
        current_season = (current_month % 12 + 3) // 3
        total_pollen = sum(pollen_values.values())
        max_pollen = max(pollen_values.values())
        
        # Create input dictionary
        current_values = {
            'MaxAirTempC': max_temp,
            'MinAirTempC': min_temp,
            'PrecipitationC': precipitation,
            'Lat': lat,
            'Long': long,
            'TempRange': temp_range,
            'Month': current_month,
            'Season': current_season,
            'TotalPollen': total_pollen,
            'MaxPollen': max_pollen
        }
        
        # Create prediction sequence
        sequence = create_prediction_sequence(data, features, sequence_length, current_values)
        
        if sequence is None:
            st.error("Failed to create prediction sequence. Please check the error messages above.")
            return
        
        # Make prediction
        safety_score = make_prediction(model, sequence)
        
        if safety_score is None:
            st.error("Failed to make prediction. Please check the error messages above.")
            return
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Safety Score", f"{safety_score:.2%}")
        
        with col2:
            safety_status = "Safe" if safety_score > 0.5 else "Not Safe"
            st.metric("Status", safety_status)
        
        with col3:
            confidence = safety_score if safety_score > 0.5 else 1 - safety_score
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Visualizations
        st.markdown("---")
        st.subheader("Environmental Conditions Visualization")
        
        # Create a gauge chart for safety score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safety_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Safety Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': safety_score * 100
                }
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Create a bar chart for pollen levels
        fig_pollen = px.bar(
            x=list(pollen_values.keys()),
            y=list(pollen_values.values()),
            title="Current Pollen Levels",
            labels={'x': 'Pollen Type', 'y': 'Count'}
        )
        
        st.plotly_chart(fig_pollen, use_container_width=True)
        
        # Add historical data visualization
        st.markdown("---")
        st.subheader("Historical Data Analysis")
        
        # Show last 30 days of data
        historical_data = data.tail(30)
        fig_historical = go.Figure()
        
        fig_historical.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['MaxAirTempC'],
            name='Max Temperature',
            line=dict(color='red')
        ))
        
        fig_historical.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['MinAirTempC'],
            name='Min Temperature',
            line=dict(color='blue')
        ))
        
        fig_historical.update_layout(
            title='Temperature Trends (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)'
        )
        
        st.plotly_chart(fig_historical, use_container_width=True)
        
        # Add city-specific analytics
        st.markdown("---")
        st.subheader(f"City Analytics for {selected_city}")
        
        # Create a map showing the city location
        fig_map = px.scatter_mapbox(
            city_data,
            lat='Lat',
            lon='Long',
            color='State',
            hover_name='City',
            zoom=4,
            title='City Location on Map'
        )
        
        fig_map.update_layout(mapbox_style='carto-positron')
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Display city-specific statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("State", city_info['State'])
            st.metric("Country", city_info['country'])
        
        with col2:
            st.metric("Region Code", city_info['iso2'])
            st.metric("Location", f"{city_info['Lat']:.4f}°, {city_info['Long']:.4f}°")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check if all required files and dependencies are properly installed.")

if __name__ == "__main__":
    main() 