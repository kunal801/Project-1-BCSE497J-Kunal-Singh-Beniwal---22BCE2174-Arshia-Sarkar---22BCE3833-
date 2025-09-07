import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

def aqi_analyzer(files):
    """
    Analyzes one or more AQI datasets from CSV files, trains ML models,
    and returns a comprehensive report with text and interactive graphs.
    """
    if not files:
        return "❌ Please upload at least one CSV file.", None, None, None

    all_data = []
    
    # Define gas keywords for flexible column matching
    gas_keywords = {
        'pm25': ['pm2.5', 'pm25', 'particulate_matter_2.5', 'fine_particulate'],
        'pm10': ['pm10', 'pm_10', 'particulate_matter_10', 'coarse_particulate'],
        'co': ['co', 'carbon_monoxide', 'carbonmonoxide'],
        'no2': ['no2', 'nitrogen_dioxide', 'nitrogendioxide'],
        'so2': ['so2', 'sulfur_dioxide', 'sulphur_dioxide'],
        'o3': ['o3', 'ozone'],
        'nh3': ['nh3', 'ammonia'],
        'benzene': ['benzene', 'c6h6'],
        'toluene': ['toluene'],
        'ethanol': ['ethanol'],
        'temperature': ['temperature', 'temp', 't'],
        'humidity': ['humidity', 'rh', 'relative_humidity'],
        'aqi': ['aqi', 'aqi_value', 'air_quality_index']
    }

    # Process all uploaded files
    for file in files:
        try:
            df = pd.read_csv(file.name)
            all_data.append(df)
        except Exception as e:
            return f"❌ Error reading file {file.name}: {e}", None, None, None

    # Concatenate all dataframes
    if not all_data:
        return "❌ No valid data found in uploaded files.", None, None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)

    identified_cols = {}
    lower_cols = [col.lower().strip() for col in combined_df.columns]

    for gas_type, keywords in gas_keywords.items():
        for i, col in enumerate(lower_cols):
            if any(k in col for k in keywords):
                identified_cols[gas_type] = combined_df.columns[i]
                break

    if 'aqi' not in identified_cols:
        return "❌ Could not find an AQI target column. Ensure a column like 'AQI' is present.", None, None, None

    feature_cols = [v for k, v in identified_cols.items() if k != 'aqi']

    if len(feature_cols) < 2:
        return "❌ Need at least two feature columns for training besides AQI.", None, None, None

    # Drop rows with missing AQI (target)
    df_cleaned = combined_df.dropna(subset=[identified_cols['aqi']]).copy()

    for col in feature_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    for col in feature_cols:
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)

    X = df_cleaned[feature_cols].values
    y = df_cleaned[identified_cols['aqi']].values
    y_class = (y > 150).astype(int)

    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Regression Model ---
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_scaled, y_train)
    y_pred_reg = regressor.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred_reg)
    rmse = np.sqrt(mse)

    # --- Classification Model ---
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_class_train)
    y_pred_class = classifier.predict(X_test_scaled)
    class_report = classification_report(y_class_test, y_pred_class, target_names=['Safe', 'Hazardous'])
    cm = confusion_matrix(y_class_test, y_pred_class)

    # --- Generate Report Text ---
    report_text = f"""
### ✅ Analysis Complete

**Detected Columns:**
{', '.join(feature_cols)}

---

#### 📈 Regression Model Performance (Predicting AQI)
- **Mean Squared Error (MSE):** `{mse:.2f}`
- **Root Mean Squared Error (RMSE):** `{rmse:.2f}`
- This model predicts the numerical AQI value. A lower RMSE indicates better accuracy.

---

#### 📊 Classification Model Performance (Detecting Hazardous Air)
- This model classifies the air quality as 'Safe' or 'Hazardous' (AQI > 150).
"""

    # --- Generate Graphs ---
    
    # 1. Prediction Plot (Actual vs. Predicted AQI)
    plot_df = pd.DataFrame({'Actual AQI': y_test, 'Predicted AQI': y_pred_reg})
    fig_pred = px.scatter(
        plot_df, 
        x='Actual AQI', 
        y='Predicted AQI',
        title='Actual vs. Predicted AQI Values',
        labels={'x': 'Actual AQI', 'y': 'Predicted AQI'},
        template='plotly_white'
    )
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Prediction', line=dict(color='red', dash='dash')))

    # 2. Feature Importance Plot
    importances = regressor.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
    fig_imp = px.bar(
        feature_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Feature Importance for AQI Prediction',
        template='plotly_white'
    )

    # 3. Confusion Matrix Plot
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Safe', 'Hazardous'],
        y=['Safe', 'Hazardous'],
        title="Confusion Matrix for Hazard Classification",
        color_continuous_scale='Viridis'
    )

    return report_text, class_report, fig_pred, fig_imp, fig_cm


# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🏭 Industrial Air Quality Analyzer")
    gr.Markdown("Upload your AQI dataset CSV files. The app will train machine learning models and provide a comprehensive report with performance metrics and interactive graphs. You can upload multiple files at once.")
    
    with gr.Row():
        file_input = gr.Files(label="Upload CSV Dataset(s)", file_types=['.csv'])
    
    with gr.Row():
        analyze_btn = gr.Button("Analyze Data & Train Models")

    with gr.Row():
        report_output = gr.Markdown(label="Analysis Report")

    with gr.Tab("Regression Analysis"):
        gr.Markdown("### Actual vs. Predicted AQI Values")
        gr.Markdown("This scatter plot shows how close the model's predictions are to the actual AQI values.")
        plot_pred = gr.Plot()
        gr.Markdown("### Feature Importance")
        gr.Markdown("This bar chart shows which gas sensors had the most influence on the AQI prediction.")
        plot_imp = gr.Plot()
        
    with gr.Tab("Classification Analysis"):
        gr.Markdown("### Classification Report")
        gr.Markdown("This table provides a detailed breakdown of the model's ability to classify air quality as 'Safe' or 'Hazardous'.")
        report_class = gr.Markdown(label="Classification Report")
        gr.Markdown("### Confusion Matrix")
        gr.Markdown("This heatmap visualizes the number of correct and incorrect classifications.")
        plot_cm = gr.Plot()

    analyze_btn.click(
        fn=aqi_analyzer,
        inputs=file_input,
        outputs=[report_output, report_class, plot_pred, plot_imp, plot_cm]
    )

# The following line is needed for the script to be runnable
if __name__ == "__main__":
    iface.launch()