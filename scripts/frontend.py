"""
🌍 AQI PREDICTION APP - GOOGLE COLAB VERSION
============================================
Complete working solution with beautiful Gradio interface
No deployment needed - loads models directly from Hopsworks!
"""

# ============================================================================
# STEP 1: CONFIGURATION - CHANGE YOUR API KEY HERE!
# ============================================================================

import os
os.environ["HOPSWORKS_API_KEY"] = HOPSWORKS_API_KEY

# ============================================================================
# STEP 2: INSTALL DEPENDENCIES
# ============================================================================
print("📦 Installing dependencies (this takes ~30 seconds)...")
!pip install -q gradio hopsworks hsml pandas numpy joblib scikit-learn xgboost plotly

print("✅ Installation complete!\n")

# ============================================================================
# STEP 3: IMPORT LIBRARIES
# ============================================================================
import gradio as gr
import pandas as pd
import numpy as np
import json
import joblib
import hopsworks
import plotly.graph_objects as go
from datetime import datetime

print("✅ Libraries imported!\n")

# ============================================================================
# STEP 4: LOAD MODELS FROM HOPSWORKS
# ============================================================================
print("="*70)
print("LOADING MODELS FROM HOPSWORKS MODEL REGISTRY")
print("="*70)
print("\n🔐 Connecting to Hopsworks...")

def load_model_from_hopsworks(model_name):
    """Load a single model from Hopsworks"""
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        mr = project.get_model_registry()

        print(f"📥 Loading {model_name}...")

        model_obj = mr.get_model(model_name)
        model_dir = model_obj.download()
        model_key = model_name.replace('aqi_', '')

        model = joblib.load(f"{model_dir}/{model_key}_model.pkl")
        scaler = joblib.load(f"{model_dir}/{model_key}_scaler.pkl")

        with open(f"{model_dir}/{model_key}_features.json", 'r') as f:
            features = json.load(f)['features']

        with open(f"{model_dir}/{model_key}_metrics.json", 'r') as f:
            metrics = json.load(f)

        print(f"   ✓ Loaded! R² = {metrics.get('test_r2', 0):.4f}, RMSE = {metrics.get('test_rmse', 0):.2f}")

        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metrics': metrics
        }
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

# Load all three models
print("\n🤖 Loading all 3 models...\n")

MODELS = {
    'gradient_boosting': load_model_from_hopsworks('aqi_gradient_boosting'),
    'xgboost': load_model_from_hopsworks('aqi_xgboost'),
    'random_forest': load_model_from_hopsworks('aqi_random_forest')
}

# Check if all models loaded successfully
if all(MODELS.values()):
    print("\n✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*70)
else:
    print("\n❌ FAILED TO LOAD MODELS!")
    print("Please check your API key and ensure models exist in Hopsworks")
    import sys
    sys.exit(1)

# ============================================================================
# STEP 5: PREDICTION FUNCTIONS
# ============================================================================

def get_aqi_category(aqi_value):
    """Get AQI category, color, and health information"""
    if aqi_value <= 50:
        return {
            'name': 'Good',
            'emoji': '😊',
            'color': '🟢',
            'health': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'recommendations': [
                '✅ Great day for outdoor activities!',
                '✅ Air quality is ideal for all groups',
                '✅ No health precautions needed'
            ]
        }
    elif aqi_value <= 100:
        return {
            'name': 'Moderate',
            'emoji': '😐',
            'color': '🟡',
            'health': 'Air quality is acceptable. However, there may be a risk for some people.',
            'recommendations': [
                '⚠️ Unusually sensitive people should limit prolonged outdoor exertion',
                '✅ Generally acceptable for most people',
                '⚠️ Monitor health if sensitive to air pollution'
            ]
        }
    elif aqi_value <= 150:
        return {
            'name': 'Unhealthy for Sensitive Groups',
            'emoji': '😷',
            'color': '🟠',
            'health': 'Members of sensitive groups may experience health effects.',
            'recommendations': [
                '⚠️ Sensitive groups should reduce prolonged outdoor exertion',
                '⚠️ Children, elderly, and people with respiratory conditions be cautious',
                '💡 Consider moving activities indoors'
            ]
        }
    elif aqi_value <= 200:
        return {
            'name': 'Unhealthy',
            'emoji': '😨',
            'color': '🔴',
            'health': 'Everyone may begin to experience health effects.',
            'recommendations': [
                '🚫 Everyone should reduce prolonged outdoor exertion',
                '🚫 Sensitive groups should avoid outdoor activities',
                '💡 Keep windows closed if possible'
            ]
        }
    elif aqi_value <= 300:
        return {
            'name': 'Very Unhealthy',
            'emoji': '😰',
            'color': '🟣',
            'health': 'Health alert: The risk of health effects is increased for everyone.',
            'recommendations': [
                '🚫 Everyone should avoid prolonged outdoor exertion',
                '🚫 Sensitive groups should remain indoors',
                '💡 Use air purifiers indoors if available'
            ]
        }
    else:
        return {
            'name': 'Hazardous',
            'emoji': '☠️',
            'color': '🟤',
            'health': 'Health warning of emergency conditions: everyone is more likely to be affected.',
            'recommendations': [
                '🚫 Everyone should avoid all outdoor activities',
                '🚫 Remain indoors and keep activity levels low',
                '⚠️ Follow emergency health advisories'
            ]
        }


def predict_aqi(model_choice, pm25, pm10, co, no2, o3, so2, hour, month, day_of_week):
    """
    Make AQI prediction using selected model

    Returns: gauge_chart, result_text
    """

    # Map model choice to model data
    model_map = {
        "🏆 Gradient Boosting (Best - R²: 0.94)": 'gradient_boosting',
        "⚡ XGBoost (Fast - R²: 0.83)": 'xgboost',
        "🌲 Random Forest (Stable - R²: 0.71)": 'random_forest'
    }

    model_key = model_map[model_choice]
    model_data = MODELS[model_key]

    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    metrics = model_data['metrics']

    # Prepare input data
    input_data = {
        'pm2_5': pm25,
        'pm10': pm10,
        'co': co,
        'no2': no2,
        'o3': o3,
        'so2': so2,
        'hour': hour,
        'month': month,
        'day_of_week': day_of_week,
        'pm_ratio': pm25 / (pm10 + 0.001),
        'total_pm': pm25 + pm10
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all features are present
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Make prediction
    X = input_df[features].values
    X_scaled = scaler.transform(X)
    predicted_aqi = model.predict(X_scaled)[0]

    # Get category info
    category_info = get_aqi_category(predicted_aqi)

    # Create gauge chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🌍 Air Quality Index", 'font': {'size': 28, 'color': 'darkblue'}},
        number={'font': {'size': 60, 'color': 'darkblue'}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#00e400'},
                {'range': [50, 100], 'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 6},
                'thickness': 0.75,
                'value': predicted_aqi
            }
        }
    ))

    gauge_fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        font={'size': 18, 'family': 'Arial'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Create result text with all information
    recommendations_text = '\n'.join([f"  {rec}" for rec in category_info['recommendations']])

    result_text = f"""
# 🎯 Prediction Results

## AQI Value: **{predicted_aqi:.2f}**

## Category: {category_info['color']} **{category_info['name']}** {category_info['emoji']}

---

### 💡 Health Impact
{category_info['health']}

### 📋 Recommendations
{recommendations_text}

---

### 📊 Model Performance
- **Model Used:** {model_choice}
- **R² Score:** {metrics['test_r2']:.4f} (explains {metrics['test_r2']*100:.2f}% of variance)
- **RMSE:** {metrics['test_rmse']:.2f}
- **MAE:** {metrics['test_mae']:.2f}
- **MAPE:** {metrics['test_mape']:.2f}%

### 📈 Input Summary
- **PM2.5:** {pm25} µg/m³
- **PM10:** {pm10} µg/m³
- **CO:** {co} µg/m³
- **NO2:** {no2} µg/m³
- **O3:** {o3} µg/m³
- **SO2:** {so2} µg/m³
- **Time:** Hour {hour}, Month {month}, Day {day_of_week}

*Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """

    return gauge_fig, result_text


# ============================================================================
# STEP 6: CREATE GRADIO INTERFACE
# ============================================================================

print("\n🎨 Creating Gradio interface...")

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%) !important;
    border: none !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
"""

# Create the interface
with gr.Blocks(css=custom_css, title="🌍 AQI Prediction System", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("""
    # 🌍 Air Quality Index (AQI) Prediction System
    ### Real-time AQI predictions using ML models trained on Hopsworks

    Predict air quality based on pollutant concentrations and make informed decisions about outdoor activities!
    """)

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")

            model_choice = gr.Dropdown(
                choices=[
                    "🏆 Gradient Boosting (Best - R²: 0.94)",
                    "⚡ XGBoost (Fast - R²: 0.83)",
                    "🌲 Random Forest (Stable - R²: 0.71)"
                ],
                value="🏆 Gradient Boosting (Best - R²: 0.94)",
                label="🤖 Select Model",
                info="Choose the ML model for prediction"
            )

            gr.Markdown("---")
            gr.Markdown("## 📊 Primary Pollutants")

            pm25 = gr.Slider(0, 500, value=35, step=0.1, label="PM2.5 (µg/m³)",
                            info="Fine particulate matter")
            pm10 = gr.Slider(0, 600, value=70, step=0.1, label="PM10 (µg/m³)",
                            info="Coarse particulate matter")
            co = gr.Slider(0, 5000, value=500, step=1, label="CO (µg/m³)",
                          info="Carbon monoxide")

            gr.Markdown("## 🌫️ Secondary Pollutants")

            no2 = gr.Slider(0, 200, value=40, step=0.1, label="NO2 (µg/m³)",
                           info="Nitrogen dioxide")
            o3 = gr.Slider(0, 300, value=60, step=0.1, label="O3 (µg/m³)",
                          info="Ozone")
            so2 = gr.Slider(0, 100, value=10, step=0.1, label="SO2 (µg/m³)",
                           info="Sulfur dioxide")

            gr.Markdown("---")
            gr.Markdown("## ⏰ Temporal Features")

            hour = gr.Slider(0, 23, value=12, step=1, label="Hour of Day (0-23)")
            month = gr.Slider(1, 12, value=6, step=1, label="Month (1-12)")
            day_of_week = gr.Slider(0, 6, value=3, step=1, label="Day of Week (0=Monday, 6=Sunday)")

            gr.Markdown("---")

            # Predict button
            predict_btn = gr.Button("🔮 Predict AQI", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### 📋 Quick Test Scenarios")
            gr.Markdown("*Click to load pre-configured air quality scenarios*")

            # Sample data buttons
            with gr.Row():
                good_btn = gr.Button("😊 Good", size="sm", variant="secondary")
                moderate_btn = gr.Button("😐 Moderate", size="sm", variant="secondary")
                unhealthy_btn = gr.Button("😨 Unhealthy", size="sm", variant="secondary")

            # Sample data functions
            def load_good_sample():
                return 12, 45, 400, 20, 40, 5, 14, 5, 2

            def load_moderate_sample():
                return 30, 90, 800, 50, 75, 15, 12, 7, 1

            def load_unhealthy_sample():
                return 100, 250, 2000, 120, 110, 40, 18, 12, 4

            good_btn.click(
                load_good_sample,
                outputs=[pm25, pm10, co, no2, o3, so2, hour, month, day_of_week]
            )
            moderate_btn.click(
                load_moderate_sample,
                outputs=[pm25, pm10, co, no2, o3, so2, hour, month, day_of_week]
            )
            unhealthy_btn.click(
                load_unhealthy_sample,
                outputs=[pm25, pm10, co, no2, o3, so2, hour, month, day_of_week]
            )

        # Right column - Results
        with gr.Column(scale=1):
            gr.Markdown("## 🎯 Prediction Results")

            gauge_plot = gr.Plot(label="AQI Gauge", show_label=False)
            result_markdown = gr.Markdown()

    # Connect prediction button
    predict_btn.click(
        fn=predict_aqi,
        inputs=[model_choice, pm25, pm10, co, no2, o3, so2, hour, month, day_of_week],
        outputs=[gauge_plot, result_markdown]
    )

    # Footer
    gr.Markdown("""
    ---
    ### 📖 Understanding AQI Categories

    | Range | Category | Color | Health Impact |
    |-------|----------|-------|---------------|
    | 0-50 | Good | 🟢 Green | Satisfactory, no risk |
    | 51-100 | Moderate | 🟡 Yellow | Acceptable for most |
    | 101-150 | Unhealthy for Sensitive | 🟠 Orange | Sensitive groups affected |
    | 151-200 | Unhealthy | 🔴 Red | Everyone may be affected |
    | 201-300 | Very Unhealthy | 🟣 Purple | Health alert |
    | 301-500 | Hazardous | 🟤 Maroon | Emergency conditions |

    ---

    **Model Information:**
    - 🏆 **Gradient Boosting**: Best accuracy (R²=0.9445), recommended for critical decisions
    - ⚡ **XGBoost**: Fast inference (R²=0.8293), great for real-time applications
    - 🌲 **Random Forest**: Most stable (R²=0.7132), reliable baseline

    *All models trained on real air quality data and validated using cross-validation*
    """)

print("✅ Gradio interface created!\n")

# ============================================================================
# STEP 7: LAUNCH THE APP
# ============================================================================

print("="*70)
print("🚀 LAUNCHING AQI PREDICTION APP")
print("="*70)
print("\n✨ Your app is starting...")
print("📱 A public URL will appear below in ~5 seconds")
print("🌐 Share this URL with anyone to use your AQI predictor!")
print("\n💡 Tip: The app stays alive as long as this cell is running")
print("="*70 + "\n")

# Launch with public link
demo.launch(
    share=True,  # Creates public URL
    debug=False,
    show_error=True,
    server_name="0.0.0.0",
    server_port=7860
)
