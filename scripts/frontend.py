HOPSWORKS_API_KEY = "GqA1WvugxU2aQHq6.1bJMxQU6Urym9vW74s5v6r2xungp1QU3MaiWKB0JrcN4JX5mjfLSry2fqu0OWYe7"



import os

if not HOPSWORKS_API_KEY:
    raise ValueError(" ERROR: HOPSWORKS_API_KEY not found in environment.\n"
                     " Set it using:\n"
                     "   export HOPSWORKS_API_KEY=your_key_here")

os.environ["HOPSWORKS_API_KEY"] = HOPSWORKS_API_KEY


import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_models_from_hopsworks():
    """Load all trained models from Hopsworks Model Registry"""
    print("\n" + "="*70)
    print("LOADING MODELS FROM HOPSWORKS")
    print("="*70)
    
    try:
        import hopsworks
        
      
        try:
            import xgboost
            print("\n XGBoost already installed")
        except ImportError:
            print("\n XGBoost not found. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', '-q', 'xgboost'])
            print(" XGBoost installed successfully")
        
        print("\nConnecting to Hopsworks...")
        project = hopsworks.login(
            project="mehveenf",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()
        
        models = {}
        model_names = ['aqi_gradient_boosting', 'aqi_random_forest', 'aqi_xgboost']
        
        for model_name in model_names:
            try:
                print(f"\n Loading {model_name}...")
                model = mr.get_model(model_name, version=1)
                model_dir = model.download()
                
                # Extract model type
                model_type = "_".join(model_name.split("_")[1:])
                
                # Load artifacts
                regressor = joblib.load(f"{model_dir}/{model_type}_model.pkl")
                scaler = joblib.load(f"{model_dir}/{model_type}_scaler.pkl")
                
                with open(f"{model_dir}/{model_type}_features.json", 'r') as f:
                    features = json.load(f)['features']
                
                with open(f"{model_dir}/{model_type}_metrics.json", 'r') as f:
                    metrics = json.load(f)
                
                models[model_type] = {
                    'regressor': regressor,
                    'scaler': scaler,
                    'features': features,
                    'metrics': metrics,
                    'name': model_name
                }
                
                print(f"Loaded {model_name}")
                print(f"   R² Score: {metrics['test_r2']:.4f}")
                print(f"   RMSE: {metrics['test_rmse']:.2f}")
                print(f"   MAE: {metrics['test_mae']:.2f}")
                
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")
                continue
        
        if not models:
            raise Exception("No models loaded successfully!")
        
        # Find best model
        best_model = max(models.items(), key=lambda x: x[1]['metrics']['test_r2'])
        best_name = best_model[0]
        best_r2 = best_model[1]['metrics']['test_r2']
        
        print("\n" + "="*70)
        print(f"Successfully loaded {len(models)} models")
        print(f" BEST MODEL: {best_name.replace('_', ' ').title()} (R² = {best_r2:.4f})")
        print("="*70)
        
        return models, best_name
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None



def fetch_latest_data():
    """Fetch the most recent data from Hopsworks Feature Store"""
    try:
        import hopsworks
        
        project = hopsworks.login(
            project="mehveenf",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        fs = project.get_feature_store()
        
        fg = fs.get_feature_group(name="air_quality_features", version=1)
        df = fg.read()
        
       
        df = df.sort_values('datetime_utc', ascending=False)
        latest = df.iloc[0]
        
        return latest
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None



def generate_forecast_from_today(models, latest_data, best_model_name):
    """Generate forecast from current date + next 3 days (4 days total = 96 hours)"""
    
   
    exclude_cols = ['id', 'datetime_utc', 'aqi', 'aqi_category', 'dominant_pollutant']
    exclude_cols += [col for col in latest_data.index if col.startswith('aqi_') and col != 'aqi']
    
   
    base_features = {k: v for k, v in latest_data.items() if k not in exclude_cols}
    
    
    current_time = datetime.now()
    
    print(f"\n Current Date/Time: {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f" Forecasting: Today + Next 3 Days (96 hours)")
    print(f" End Date: {(current_time + timedelta(hours=96)).strftime('%Y-%m-%d %H:%M')}")
    
    # Initialize results
    results = []
    
    
    for hour in range(96):
        forecast_time = current_time + timedelta(hours=hour)
        
        
        current_features = base_features.copy()
        current_features['month'] = float(forecast_time.month)
        if 'hour' in current_features:
            current_features['hour'] = float(forecast_time.hour)
        if 'day_of_week' in current_features:
            current_features['day_of_week'] = float(forecast_time.weekday())
        
        
        predictions = {}
        
        for model_key, model_data in models.items():
            try:
                # Create input DataFrame
                input_df = pd.DataFrame([{
                    feat: current_features.get(feat, 0.0) 
                    for feat in model_data['features']
                }])
                
                # Scale and predict
                input_scaled = model_data['scaler'].transform(input_df)
                pred = model_data['regressor'].predict(input_scaled)[0]
                pred = np.clip(pred, 0, 500)
                
                predictions[model_key] = pred
                
            except Exception as e:
                predictions[model_key] = np.nan
        
       
        valid_preds = [p for p in predictions.values() if not np.isnan(p)]
        ensemble = np.mean(valid_preds) if valid_preds else np.nan
        
       
        results.append({
            'datetime': forecast_time,
            'date': forecast_time.strftime('%Y-%m-%d'),
            'time': forecast_time.strftime('%H:%M'),
            'day_name': forecast_time.strftime('%A'),
            'hour_of_day': forecast_time.hour,
            **predictions,
            'ensemble': ensemble,
            'best_model': predictions.get(best_model_name, np.nan)
        })
    
    df = pd.DataFrame(results)
    print(f" Generated {len(df)} hourly predictions")
    print(f"   From: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')}")
    print(f"   To:   {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    
    return df



def create_forecast_visualization(df_forecast, models, best_model_name):
    """Create comprehensive forecast visualization with model comparison"""
    
   
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            '4-Day AQI Forecast (Today + Next 3 Days)',
            'Model Performance Comparison'
        ),
        vertical_spacing=0.12
    )
    
    
    colors = {
        'gradient_boosting': '#1f77b4',
        'random_forest': '#ff7f0e',
        'xgboost': '#2ca02c',
        'ensemble': '#d62728',
        'best_model': '#9467bd'
    }
    
   
    for model_key in models.keys():
        if model_key in df_forecast.columns:
            is_best = (model_key == best_model_name)
            
            fig.add_trace(go.Scatter(
                x=df_forecast['datetime'],
                y=df_forecast[model_key],
                mode='lines',
                name=f"{model_key.replace('_', ' ').title()}" + (" " if is_best else ""),
                line=dict(
                    color=colors.get(model_key, '#999'),
                    width=4 if is_best else 2,
                    dash='solid' if is_best else 'dash'
                ),
                opacity=1.0 if is_best else 0.4,
                legendgroup=model_key,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             '%{x|%b %d, %H:%M}<br>' +
                             'AQI: <b>%{y:.1f}</b><br>' +
                             '<extra></extra>'
            ), row=1, col=1)
    
    
    if 'ensemble' in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast['datetime'],
            y=df_forecast['ensemble'],
            mode='lines',
            name='Ensemble Avg',
            line=dict(color=colors['ensemble'], width=3),
            legendgroup='ensemble',
            hovertemplate='<b>Ensemble Average</b><br>' +
                         '%{x|%b %d, %H:%M}<br>' +
                         'AQI: <b>%{y:.1f}</b><br>' +
                         '<extra></extra>'
        ), row=1, col=1)
    
   
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0, row=1, col=1)
    
    model_scores = []
    model_names = []
    model_colors_list = []
    
    for model_key, model_data in models.items():
        model_names.append(model_key.replace('_', ' ').title() + (" " if model_key == best_model_name else ""))
        model_scores.append(model_data['metrics']['test_r2'])
        model_colors_list.append(colors.get(model_key, '#999'))
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=model_scores,
        marker_color=model_colors_list,
        text=[f"R²={score:.4f}" for score in model_scores],
        textposition='outside',
        name='R² Score',
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>' +
                     'R² Score: <b>%{y:.4f}</b><br>' +
                     '<extra></extra>'
    ), row=2, col=1)
    
   
    fig.update_xaxes(title_text="Date & Time", row=1, col=1, tickformat='%b %d<br>%H:%M')
    fig.update_yaxes(title_text="AQI", row=1, col=1)
    
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="R² Score", row=2, col=1, range=[0, 1])
    
    fig.update_layout(
        height=900,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig



def create_detailed_stats(df_forecast, models, best_model_name):
    """Create detailed statistics table for ALL models"""
    
    # Get unique dates
    unique_dates = df_forecast['date'].unique()
    
    stats = []
    
    # Get all model keys
    model_keys = list(models.keys())
    
    # Overall stats for each model
    for model_key in model_keys:
        if model_key in df_forecast.columns:
            is_best = (model_key == best_model_name)
            model_display_name = model_key.replace('_', ' ').title() + (" 🏆" if is_best else "")
            
            stats.append({
                'Model': model_display_name,
                'Period': 'OVERALL (96h)',
                'Avg AQI': f"{df_forecast[model_key].mean():.1f}",
                'Max AQI': f"{df_forecast[model_key].max():.1f}",
                'Min AQI': f"{df_forecast[model_key].min():.1f}",
                'Std Dev': f"{df_forecast[model_key].std():.1f}"
            })
    
    # Add ensemble overall
    if 'ensemble' in df_forecast.columns:
        stats.append({
            'Model': 'Ensemble Avg',
            'Period': 'OVERALL (96h)',
            'Avg AQI': f"{df_forecast['ensemble'].mean():.1f}",
            'Max AQI': f"{df_forecast['ensemble'].max():.1f}",
            'Min AQI': f"{df_forecast['ensemble'].min():.1f}",
            'Std Dev': f"{df_forecast['ensemble'].std():.1f}"
        })
    
    # Add separator row
    stats.append({
        'Model': '─' * 20,
        'Period': '─' * 15,
        'Avg AQI': '─' * 8,
        'Max AQI': '─' * 8,
        'Min AQI': '─' * 8,
        'Std Dev': '─' * 8
    })
    
    # Daily stats for each model
    for date in unique_dates:
        day_data = df_forecast[df_forecast['date'] == date]
        day_name = day_data.iloc[0]['day_name']
        period_label = f"{date} ({day_name})"
        
        for model_key in model_keys:
            if model_key in df_forecast.columns:
                is_best = (model_key == best_model_name)
                model_display_name = model_key.replace('_', ' ').title() + (" " if is_best else "")
                
                stats.append({
                    'Model': model_display_name,
                    'Period': period_label,
                    'Avg AQI': f"{day_data[model_key].mean():.1f}",
                    'Max AQI': f"{day_data[model_key].max():.1f}",
                    'Min AQI': f"{day_data[model_key].min():.1f}",
                    'Std Dev': f"{day_data[model_key].std():.1f}"
                })
        
        # Add ensemble for this day
        if 'ensemble' in df_forecast.columns:
            stats.append({
                'Model': 'Ensemble Avg',
                'Period': period_label,
                'Avg AQI': f"{day_data['ensemble'].mean():.1f}",
                'Max AQI': f"{day_data['ensemble'].max():.1f}",
                'Min AQI': f"{day_data['ensemble'].min():.1f}",
                'Std Dev': f"{day_data['ensemble'].std():.1f}"
            })
        
        # Add separator after each day
        if date != unique_dates[-1]:  # Don't add separator after last day
            stats.append({
                'Model': '─' * 20,
                'Period': '─' * 15,
                'Avg AQI': '─' * 8,
                'Max AQI': '─' * 8,
                'Min AQI': '─' * 8,
                'Std Dev': '─' * 8
            })
    
    return pd.DataFrame(stats)



def create_model_comparison_table(models, best_model_name):
    """Create model performance comparison table"""
    
    comparison = []
    
    for model_key, model_data in models.items():
        metrics = model_data['metrics']
        is_best = (model_key == best_model_name)
        
        comparison.append({
            'Model': model_key.replace('_', ' ').title() + ("  BEST" if is_best else ""),
            'R² Score': f"{metrics['test_r2']:.4f}",
            'RMSE': f"{metrics['test_rmse']:.2f}",
            'MAE': f"{metrics['test_mae']:.2f}",
            'MAPE (%)': f"{metrics['test_mape']:.2f}",
            'Accuracy': f"{metrics['test_r2']*100:.2f}%"
        })
    
    # Sort by R² descending
    df = pd.DataFrame(comparison)
    df = df.sort_values('R² Score', ascending=False)
    
    return df


def create_gradio_dashboard(models, best_model_name):
    """Create comprehensive Gradio dashboard"""
    
    def generate_forecast():
        """Generate and display forecast"""
        try:
           
            print("\n Fetching latest data...")
            latest_data = fetch_latest_data()
            if latest_data is None:
                return None, None, None, " Failed to fetch data from Hopsworks"
            
            print(f" Latest data from Feature Store: {latest_data['datetime_utc']}")
            
            # Generate forecast from TODAY
            print("\n Generating forecast from current date...")
            df_forecast = generate_forecast_from_today(models, latest_data, best_model_name)
            
            # Create visualizations
            fig = create_forecast_visualization(df_forecast, models, best_model_name)
            stats_df = create_detailed_stats(df_forecast, models, best_model_name)
            model_comparison_df = create_model_comparison_table(models, best_model_name)
            
            # Create summary text
            current_time = datetime.now()
            end_time = current_time + timedelta(hours=96)
            
            best_model_metrics = models[best_model_name]['metrics']
            
            summary = f"""
#  Forecast Summary

##  Time Period
- **Current Date/Time:** {current_time.strftime('%A, %B %d, %Y at %H:%M')}
- **Forecast Coverage:** Today + Next 3 Days (4 days total)
- **End Date/Time:** {end_time.strftime('%A, %B %d, %Y at %H:%M')}
- **Total Hours:** 96 hours

---

##  Best Model Selected
**{best_model_name.replace('_', ' ').title()}**

### Performance Metrics:
- **R² Score:** {best_model_metrics['test_r2']:.4f} (explains {best_model_metrics['test_r2']*100:.2f}% of variance)
- **RMSE:** {best_model_metrics['test_rmse']:.2f}
- **MAE:** {best_model_metrics['test_mae']:.2f}
- **MAPE:** {best_model_metrics['test_mape']:.2f}%

This is the most accurate model based on test set performance.

---

## Key Predictions (Using Best Model)
- **Average AQI (96h):** {df_forecast['best_model'].mean():.1f}
- **Peak AQI:** {df_forecast['best_model'].max():.1f} at {df_forecast.loc[df_forecast['best_model'].idxmax(), 'datetime'].strftime('%b %d, %H:%M')}
- **Best AQI:** {df_forecast['best_model'].min():.1f} at {df_forecast.loc[df_forecast['best_model'].idxmin(), 'datetime'].strftime('%b %d, %H:%M')}

---

*Dashboard updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Data source: Hopsworks Feature Store*
            """
            
            return fig, stats_df, model_comparison_df, summary
            
        except Exception as e:
            import traceback
            error_msg = f" Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, None, None, error_msg
    
    # Create interface
    with gr.Blocks(title="4-Day AQI Forecast", theme=gr.themes.Soft()) as dashboard:
        
        gr.Markdown(f"""
        #  Air Quality Forecast Dashboard
        ### Karachi, Pakistan - Today + Next 3 Days
        
        Real-time predictions using the **best performing model**: **{best_model_name.replace('_', ' ').title()} **
        """)
        
        with gr.Row():
            refresh_btn = gr.Button(" Generate Forecast from Today", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        with gr.Row():
            summary_text = gr.Markdown()
        
        gr.Markdown("---")
        
        forecast_plot = gr.Plot(label="AQI Forecast & Model Comparison")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("###  Daily Statistics (Best Model)")
                stats_table = gr.Dataframe(label="", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("###  Model Performance Comparison")
                model_table = gr.Dataframe(label="", interactive=False)
        
        gr.Markdown("""
        ---
        ### AQI Categories
        
        | Range | Category | Health Impact |
        |-------|----------|---------------|
        | 0-50 | 🟢 Good | Air quality is satisfactory |
        | 51-100 | 🟡 Moderate | Acceptable for most people |
        | 101-150 | 🟠 Unhealthy (Sensitive) | Sensitive groups affected |
        | 151-200 | 🔴 Unhealthy | Everyone may experience effects |
        | 201-300 | 🟣 Very Unhealthy | Health alert |
        | 301-500 | 🟤 Hazardous | Emergency conditions |
        """)
        
        # Auto-load on page load
        dashboard.load(
            fn=generate_forecast,
            inputs=None,
            outputs=[forecast_plot, stats_table, model_table, summary_text]
        )
        
        # Refresh button
        refresh_btn.click(
            fn=generate_forecast,
            inputs=None,
            outputs=[forecast_plot, stats_table, model_table, summary_text]
        )
    
    return dashboard


if __name__ == "__main__":
    
  
    
    # Check API key
    if not os.getenv("HOPSWORKS_API_KEY"):
        print(" ERROR: HOPSWORKS_API_KEY not set!")
        exit(1)
    
    # Load models and identify best one
    models, best_model_name = load_models_from_hopsworks()
    
    if not models:
        print(" Failed to load models. Exiting...")
        exit(1)
    
    # Create dashboard
    print("\n Creating dashboard...")
    dashboard = create_gradio_dashboard(models, best_model_name)
    print(" Dashboard created!")
    
    # Launch
    print("\n" + "="*70)
    print(" LAUNCHING FORECAST DASHBOARD")
    print("="*70)
    print(f"\n Best Model: {best_model_name.replace('_', ' ').title()}")
    print(" Forecast: Today + Next 3 Days (96 hours)")
    print(" Public URL will appear below")
    print("="*70 + "\n")
    
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
