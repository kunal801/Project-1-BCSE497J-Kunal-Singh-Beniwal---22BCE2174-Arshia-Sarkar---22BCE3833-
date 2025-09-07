# industrial_aqi_ui.py
import streamlit as st
import pandas as pd
from industrial_aqi_system import IndustrialAQIModel  # Assuming the above code saved as industrial_aqi_system.py

st.set_page_config(page_title="Industrial AQI Prediction", layout="wide")

@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = IndustrialAQIModel()
    # Load pre-trained model if available, else train
    try:
        model.load('aqi_model.pkl')
        st.info("Loaded existing trained model.")
    except:
        st.info("Training model (this might take a minute)...")
        data = model.generate_sample_data(days=90)
        model.train(data)
        model.save()
        st.success("Training complete. Model saved.")
    # Load or generate data for prediction context
    try:
        df = pd.read_csv('industrial_aqi_sample_data.csv', parse_dates=['timestamp'])
        st.info("Loaded existing sample data.")
    except:
        df = model.generate_sample_data(days=90)
        df.to_csv('industrial_aqi_sample_data.csv', index=False)
        st.info("Generated new sample data.")
    return model, df

def main():
    st.title("Industrial Air Quality Index (AQI) Prediction System")

    model, data = load_model_and_data()

    st.sidebar.header("User Inputs")

    # Date selector
    date = st.sidebar.date_input("Select Date", value=pd.to_datetime("2025-02-15"))
    # Time selector
    time = st.sidebar.time_input("Select Time", value=pd.to_datetime("10:30").time())
    # Combine into datetime
    selected_datetime = pd.Timestamp.combine(date, time)

    st.write(f"### Prediction for {selected_datetime}")

    # Predict using the model
    prediction = model.predict(data, selected_datetime)

    if prediction is None:
        st.error("Not enough data to predict for this datetime. Please choose a different time.")
        return

    # Show overall AQI status
    st.write(f"**Overall AQI Level:** {prediction['overall_aqi']}")

    # Show environmental conditions
    env = prediction.get('environmental_conditions', {})
    if env:
        st.write("**Environmental Conditions:**")
        st.write(f"- Temperature: {env.get('temperature', 'N/A')} °C")
        st.write(f"- Humidity: {env.get('humidity', 'N/A')} %")
        st.write(f"- Wind Speed: {env.get('wind_speed', 'N/A')} m/s")
        st.write(f"- Production Activity: {env.get('production_activity', 'N/A')}")

    # Show individual gas predictions with safety levels
    st.write("#### Gas Concentrations and Safety Levels")
    safety_color = {"SAFE": "green", "CAUTION": "orange", "DANGER": "red"}
    for gas, results in prediction['predictions'].items():
        level = results['lvl']
        value = results['pred']
        st.markdown(
            f"<span style='color: {safety_color.get(level,'black')}'>{gas}: {value} - {level}</span>",
            unsafe_allow_html=True)

if __name__ == "__main__":
    main()

