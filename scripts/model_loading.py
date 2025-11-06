"""
Complete AQI Model Training Script
===================================
Fetches data from Hopsworks, trains 3 ML models, and registers them
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: FETCH DATA FROM HOPSWORKS
# ============================================================================

def fetch_data_from_hopsworks(feature_group_name="air_quality_features", version=1):
    """
    Fetch data from Hopsworks Feature Store
    """
    print("\n" + "="*70)
    print("FETCHING DATA FROM HOPSWORKS")
    print("="*70)

    try:
        import hopsworks

        print("\n🔐 Connecting to Hopsworks...")
        project = hopsworks.login(
            project="mehveenf",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        print(f"   ✓ Connected to project: {project.name}")

        fs = project.get_feature_store()

        print(f"\n📥 Fetching feature group: {feature_group_name} (v{version})")
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)

        df = feature_group.read()

        print(f"\n✅ DATA RETRIEVED SUCCESSFULLY!")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {len(df.columns)}")
        print("="*70)

        return df

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("   Please check your API key and feature group name")
        return None


# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================

def prepare_data(df, target='aqi'):
    """
    Prepare data for model training
    """
    print("\n" + "="*70)
    print("PREPARING DATA FOR TRAINING")
    print("="*70)

    # Check if target exists
    if target not in df.columns:
        print(f"❌ ERROR: Target column '{target}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Columns to exclude from features
    exclude_cols = [target, 'id', 'datetime_utc', 'aqi_category', 'dominant_pollutant']

    # Also exclude individual pollutant AQI columns if they exist
    aqi_cols = [col for col in df.columns if col.startswith('aqi_') and col != target]
    exclude_cols.extend(aqi_cols)

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n📊 Dataset Information:")
    print(f"   • Total rows: {len(df)}")
    print(f"   • Feature columns: {len(feature_cols)}")
    print(f"   • Target column: {target}")

    # Handle categorical features
    df_features = df[feature_cols].copy()
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        print(f"\n🔄 Encoding {len(categorical_cols)} categorical features...")
        df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
        feature_cols = df_features.columns.tolist()
        print(f"   ✓ Features after encoding: {len(feature_cols)}")

    # Prepare X and y
    X = df_features.values
    y = df[target].values

    print(f"\n   • Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   • Target mean: {y.mean():.2f}")

    # Train/Val/Test split (70/10/20)
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
    )

    print(f"   • Training:   {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Validation: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   • Test:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Feature scaling
    print(f"\n⚖️  Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print("   ✓ Features scaled")

    print("\n✅ DATA PREPARATION COMPLETE")
    print("="*70)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_cols
    }


# ============================================================================
# STEP 3: TRAIN MODELS
# ============================================================================

def train_models(data):
    """
    Train 3 best ML models for AQI prediction
    """
    print("\n" + "="*70)
    print("TRAINING 3 ML MODELS")
    print("="*70)

    results = {}

    # Model 1: Random Forest
    print("\n📊 MODEL 1: Random Forest Regressor")
    print("-" * 70)
    start = datetime.now()

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(data['X_train'], data['y_train'])

    rf_train_pred = rf_model.predict(data['X_train'])
    rf_val_pred = rf_model.predict(data['X_val'])
    rf_test_pred = rf_model.predict(data['X_test'])

    results['random_forest'] = {
        'model': rf_model,
        'name': 'Random Forest Regressor',
        'train_rmse': np.sqrt(mean_squared_error(data['y_train'], rf_train_pred)),
        'train_mae': mean_absolute_error(data['y_train'], rf_train_pred),
        'train_r2': r2_score(data['y_train'], rf_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(data['y_val'], rf_val_pred)),
        'val_mae': mean_absolute_error(data['y_val'], rf_val_pred),
        'val_r2': r2_score(data['y_val'], rf_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(data['y_test'], rf_test_pred)),
        'test_mae': mean_absolute_error(data['y_test'], rf_test_pred),
        'test_r2': r2_score(data['y_test'], rf_test_pred),
        'test_mape': mean_absolute_percentage_error(data['y_test'], rf_test_pred) * 100,
        'training_time': (datetime.now() - start).total_seconds()
    }

    print(f"✓ Training time: {results['random_forest']['training_time']:.2f}s")
    print(f"✓ Test R²: {results['random_forest']['test_r2']:.4f}")
    print(f"✓ Test RMSE: {results['random_forest']['test_rmse']:.4f}")

    # Model 2: Gradient Boosting
    print("\n📊 MODEL 2: Gradient Boosting Regressor")
    print("-" * 70)
    start = datetime.now()

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.9,
        random_state=42
    )
    gb_model.fit(data['X_train'], data['y_train'])

    gb_train_pred = gb_model.predict(data['X_train'])
    gb_val_pred = gb_model.predict(data['X_val'])
    gb_test_pred = gb_model.predict(data['X_test'])

    results['gradient_boosting'] = {
        'model': gb_model,
        'name': 'Gradient Boosting Regressor',
        'train_rmse': np.sqrt(mean_squared_error(data['y_train'], gb_train_pred)),
        'train_mae': mean_absolute_error(data['y_train'], gb_train_pred),
        'train_r2': r2_score(data['y_train'], gb_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(data['y_val'], gb_val_pred)),
        'val_mae': mean_absolute_error(data['y_val'], gb_val_pred),
        'val_r2': r2_score(data['y_val'], gb_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(data['y_test'], gb_test_pred)),
        'test_mae': mean_absolute_error(data['y_test'], gb_test_pred),
        'test_r2': r2_score(data['y_test'], gb_test_pred),
        'test_mape': mean_absolute_percentage_error(data['y_test'], gb_test_pred) * 100,
        'training_time': (datetime.now() - start).total_seconds()
    }

    print(f"✓ Training time: {results['gradient_boosting']['training_time']:.2f}s")
    print(f"✓ Test R²: {results['gradient_boosting']['test_r2']:.4f}")
    print(f"✓ Test RMSE: {results['gradient_boosting']['test_rmse']:.4f}")

    # Model 3: XGBoost
    print("\n📊 MODEL 3: XGBoost Regressor")
    print("-" * 70)
    start = datetime.now()

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(data['X_train'], data['y_train'])

    xgb_train_pred = xgb_model.predict(data['X_train'])
    xgb_val_pred = xgb_model.predict(data['X_val'])
    xgb_test_pred = xgb_model.predict(data['X_test'])

    results['xgboost'] = {
        'model': xgb_model,
        'name': 'XGBoost Regressor',
        'train_rmse': np.sqrt(mean_squared_error(data['y_train'], xgb_train_pred)),
        'train_mae': mean_absolute_error(data['y_train'], xgb_train_pred),
        'train_r2': r2_score(data['y_train'], xgb_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(data['y_val'], xgb_val_pred)),
        'val_mae': mean_absolute_error(data['y_val'], xgb_val_pred),
        'val_r2': r2_score(data['y_val'], xgb_val_pred),
        'test_rmse': np.sqrt(mean_squared_error(data['y_test'], xgb_test_pred)),
        'test_mae': mean_absolute_error(data['y_test'], xgb_test_pred),
        'test_r2': r2_score(data['y_test'], xgb_test_pred),
        'test_mape': mean_absolute_percentage_error(data['y_test'], xgb_test_pred) * 100,
        'training_time': (datetime.now() - start).total_seconds()
    }

    print(f"✓ Training time: {results['xgboost']['training_time']:.2f}s")
    print(f"✓ Test R²: {results['xgboost']['test_r2']:.4f}")
    print(f"✓ Test RMSE: {results['xgboost']['test_rmse']:.4f}")

    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON (Test Set)")
    print("="*70)

    comparison = pd.DataFrame([
        {
            'Model': results[k]['name'],
            'R² Score': results[k]['test_r2'],
            'RMSE': results[k]['test_rmse'],
            'MAE': results[k]['test_mae'],
            'MAPE (%)': results[k]['test_mape'],
            'Time (s)': results[k]['training_time']
        }
        for k in results.keys()
    ]).sort_values('R² Score', ascending=False)

    print("\n" + comparison.to_string(index=False))

    best_model = comparison.iloc[0]['Model']
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   R² = {comparison.iloc[0]['R² Score']:.4f}")
    print(f"   RMSE = {comparison.iloc[0]['RMSE']:.4f}")

    print("\n✅ ALL MODELS TRAINED!")
    print("="*70)

    return results, comparison


# ============================================================================
# STEP 4: REGISTER MODELS IN HOPSWORKS
# ============================================================================

def register_models(results, data):
    """
    Register all trained models in Hopsworks Model Registry
    """
    print("\n" + "="*70)
    print("REGISTERING MODELS IN HOPSWORKS")
    print("="*70)

    try:
        import hopsworks
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema

        # Connect to Hopsworks
        print("\n🔐 Connecting to Hopsworks...")
        project = hopsworks.login(
            project="mehveenf",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()

        registered = {}

        for model_key, model_data in results.items():
            print(f"\n📤 Registering: {model_data['name']}")
            print("-" * 70)

            # Create model directory
            model_dir = f"models/{model_key}"
            os.makedirs(model_dir, exist_ok=True)

            # Save model artifacts
            joblib.dump(model_data['model'], f"{model_dir}/{model_key}_model.pkl")
            joblib.dump(data['scaler'], f"{model_dir}/{model_key}_scaler.pkl")

            with open(f"{model_dir}/{model_key}_features.json", 'w') as f:
                json.dump({'features': data['feature_names']}, f)

            with open(f"{model_dir}/{model_key}_metrics.json", 'w') as f:
                json.dump({
                    'test_rmse': model_data['test_rmse'],
                    'test_mae': model_data['test_mae'],
                    'test_r2': model_data['test_r2'],
                    'test_mape': model_data['test_mape']
                }, f, indent=2)

            print(f"   ✓ Saved artifacts to {model_dir}/")

            # Create model schema with proper format (list of dicts with 'name' and 'type')
            input_schema_dict = [{"name": str(feat), "type": "double"} for feat in data['feature_names']]
            output_schema_dict = [{"name": "aqi", "type": "double"}]

            input_schema = Schema(input_schema_dict)
            output_schema = Schema(output_schema_dict)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

            # Register in Hopsworks
            print(f"   ⏳ Uploading to Hopsworks Model Registry...")
            aqi_model = mr.python.create_model(
                name=f"aqi_{model_key}",
                metrics={
                    "test_rmse": float(model_data['test_rmse']),
                    "test_mae": float(model_data['test_mae']),
                    "test_r2": float(model_data['test_r2']),
                    "test_mape": float(model_data['test_mape'])
                },
                model_schema=model_schema,
                description=f"AQI Prediction using {model_data['name']} | "
                           f"R²={model_data['test_r2']:.4f}, RMSE={model_data['test_rmse']:.4f}"
            )

            aqi_model.save(model_dir)

            print(f"   ✓ Registered in Hopsworks as 'aqi_{model_key}'")
            print(f"   ✓ Version: {aqi_model.version}")

            registered[model_key] = aqi_model

        print("\n✅ ALL MODELS REGISTERED SUCCESSFULLY!")
        print("="*70)

        return registered

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("""
╔════════════════════════════════════════════════════════════════════╗
║                  AQI PREDICTION MODEL TRAINING                     ║
║                                                                    ║
║  This script will:                                                ║
║  1. ✓ Fetch data from Hopsworks Feature Store                     ║
║  2. ✓ Prepare and split data (70/10/20)                          ║
║  3. ✓ Train 3 ML models (RF, GBM, XGBoost)                       ║
║  4. ✓ Register models in Hopsworks Model Registry                ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # Check API key
    if not os.getenv("HOPSWORKS_API_KEY"):
        print("❌ ERROR: HOPSWORKS_API_KEY not set!")
        print("   Set it with: os.environ['HOPSWORKS_API_KEY'] = 'your_key'")
        exit(1)

    # Step 1: Fetch data
    df = fetch_data_from_hopsworks(
        feature_group_name="air_quality_features",
        version=1
    )

    if df is None:
        print("❌ Failed to fetch data. Exiting...")
        exit(1)

    # Step 2: Prepare data
    data = prepare_data(df, target='aqi')

    if data is None:
        print("❌ Failed to prepare data. Exiting...")
        exit(1)

    # Step 3: Train models
    results, comparison = train_models(data)

    # Step 4: Register models
    registered_models = register_models(results, data)

    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\n✅ Summary:")
    print(f"   • Data fetched: {len(df)} rows")
    print(f"   • Features used: {len(data['feature_names'])}")
    print(f"   • Models trained: {len(results)}")
    print(f"   • Models registered: {len(registered_models) if registered_models else 0}")

    print("\n📊 Model Performance (Test Set):")
    print(comparison.to_string(index=False))

    print("\n" + "="*70)
    print("Next steps:")
    print("1. Go to your Hopsworks project")
    print("2. Navigate to Model Registry")
    print("3. View your registered models: aqi_random_forest, aqi_gradient_boosting, aqi_xgboost")
    print("4. Deploy the best model for predictions!")
    print("="*70)
