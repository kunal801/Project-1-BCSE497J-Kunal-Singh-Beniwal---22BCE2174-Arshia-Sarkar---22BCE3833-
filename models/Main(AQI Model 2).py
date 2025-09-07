import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split

def aqi_analyzer(files):
    """
    Analyzes one or more AQI datasets from CSV files and returns a report
    with a plot that is random but clustered near the ideal prediction line.
    """
    if not files:
        return "Please upload at least one CSV file.", None, None, None, None

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
            return f"Error reading file {file.name}: {e}", None, None, None, None

    if not all_data:
        return "❌ No valid data found in uploaded files.", None, None, None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    identified_cols = {}
    lower_cols = [col.lower().strip() for col in combined_df.columns]

    for gas_type, keywords in gas_keywords.items():
        for i, col in enumerate(lower_cols):
            if any(k in col for k in keywords):
                identified_cols[gas_type] = combined_df.columns[i]
                break

    if 'aqi' not in identified_cols:
        return "Could not find an AQI target column. Ensure a column like 'AQI' is present.", None, None, None, None
        
    aqi_col_name = identified_cols['aqi']
    df_cleaned = combined_df.dropna(subset=[aqi_col_name]).copy()
    y = pd.to_numeric(df_cleaned[aqi_col_name], errors='coerce').dropna()

    if y.empty:
        return "No valid AQI data found in the dataset.", None, None, None, None
    
    # Split the data just to get a test set of actual values
    _, y_test = train_test_split(y, test_size=0.2, random_state=42)

    # Generate "less accurate" predictions by adding Gaussian noise to the actual values
    # The standard deviation controls the spread of the points
    noise_std = (y_test.max() - y_test.min()) * 0.05  # 5% of the range
    noise = np.random.normal(0, noise_std, y_test.shape)
    y_pred_noisy = y_test + noise

    # --- Generate Plotly Graphs ---
    plot_df = pd.DataFrame({'Actual AQI': y_test, 'Predicted AQI': y_pred_noisy})
    fig_pred = px.scatter(
        plot_df,
        x='Actual AQI',
        y='Predicted AQI',
        title='Actual vs. "Less Accurate" Predicted AQI Values',
        labels={'x': 'Actual AQI', 'y': 'Predicted AQI'},
        template='plotly_white'
    )
    fig_pred.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Prediction', line=dict(color='red', dash='dash')))
    
    report_text = f"""
### Analysis Complete

**Plotting a "Less Accurate" Scatter Plot**
- This plot shows points clustered near the ideal line, simulating a model that is somewhat accurate but not perfect.
- The randomness of the points is controlled to be near the true values.
"""
    
    
    
    
    
    
    
    
    
    
    # Return values for Gradio's outputs (report, classification report, plots)
    return report_text, None, fig_pred, None, None

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Smart Environmental Sensing for Industrial Air Quality Analyzer")
    gr.Markdown("Upload your AQI dataset CSV files. This app will provide a comprehensive report with a plot that is intentionally random but still follows a trend.")

    with gr.Row():
        file_input = gr.Files(label="Upload CSV Dataset(s)", file_types=['.csv'])

    with gr.Row():
        analyze_btn = gr.Button("Analyze Data & Generate Plot")

    with gr.Row():
        report_output = gr.Markdown(label="Analysis Report")

    with gr.Tab("Random Plot Analysis"):
        gr.Markdown("### Actual vs. 'Less Accurate' Predicted AQI Values")
        gr.Markdown("This scatter plot shows the relationship between actual AQI values and randomly perturbed numbers.")
        plot_pred = gr.Plot()
        
    # The other tabs are now empty as we are only generating the random plot
    with gr.Tab("Classification Analysis"):
        gr.Markdown("This section is not used in this version of the analysis.")
        report_class = gr.Markdown(label="Classification Report")
        plot_cm = gr.Plot()
    
    with gr.Tab("Other Plots"):
        gr.Markdown("This section is not used in this version of the analysis.")
        plot_imp = gr.Plot()

    analyze_btn.click(
        fn=aqi_analyzer,
        inputs=file_input,
        outputs=[report_output, report_class, plot_pred, plot_imp, plot_cm]
    )

if __name__ == "__main__":
    iface.launch()