import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data from CSV
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert business_date to datetime
    df['business_date'] = pd.to_datetime(df['business_date'])
    
    # Extract date features
    df['year'] = df['business_date'].dt.year
    df['month'] = df['business_date'].dt.month
    df['quarter'] = df['business_date'].dt.quarter
    
    # Handle missing values in numerical columns
    df['ledger_amount'] = df['ledger_amount'].fillna(df['ledger_amount'].median())
    
    return df

# Function to create time-based features
def create_time_features(df):
    """
    Create time-based features for time series forecasting
    """
    # Create lag features (previous quarter's values)
    df_with_features = df.copy()
    
    # Sort by date
    df_with_features = df_with_features.sort_values('business_date')
    
    # Group by categorical columns and create lag features
    categorical_cols = ['currency', 'custom_1', 'custom_2', 'custom_3', 'account_description']
    available_cats = [col for col in categorical_cols if col in df_with_features.columns]
    
    # If we have enough data, create lag features by group
    if len(df_with_features) > 1:
        for cat in available_cats:
            groups = df_with_features.groupby(cat)
            
            # Create lag features
            for group_name, group_data in groups:
                group_data = group_data.sort_values('business_date')
                
                # Add lag of ledger_amount
                mask = df_with_features[cat] == group_name
                df_with_features.loc[mask, 'ledger_amount_lag1'] = group_data['ledger_amount'].shift(1)
        
        # Fill NaN values in lag columns with mean
        df_with_features['ledger_amount_lag1'] = df_with_features['ledger_amount_lag1'].fillna(
            df_with_features['ledger_amount_lag1'].mean())
    
    return df_with_features

# Function to prepare train/test sets
def prepare_train_test(df, target_date):
    """
    Split data into training and test sets based on date
    """
    # Convert target_date to datetime if it's a string
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Training data: all data before target_date
    train = df[df['business_date'] < target_date].copy()
    
    # Test data: the target month
    # For forecasting September, we would use the record template from June but without the target
    latest_date = train['business_date'].max()
    test_template = df[df['business_date'] == latest_date].copy()
    
    # Modify the date to be the target_date
    test_template['business_date'] = target_date
    test_template['year'] = target_date.year
    test_template['month'] = target_date.month
    test_template['quarter'] = target_date.quarter
    
    # Remove the target from the test set
    if 'ledger_amount' in test_template.columns:
        test_template['ledger_amount'] = np.nan
    
    # Features and target for training
    X_train = train.drop(['ledger_amount', 'business_date'], axis=1)
    y_train = train['ledger_amount']
    
    # Features for test/prediction
    X_test = test_template.drop(['ledger_amount', 'business_date'], axis=1)
    
    return X_train, y_train, X_test, test_template

# Function to build and evaluate models
def build_and_evaluate_models(X_train, y_train, categorical_features):
    """
    Build and evaluate multiple regression models
    """
    # Create preprocessor for mixed data types
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Identify numeric features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    # Store results
    results = {}
    fitted_models = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=tscv, 
            scoring='neg_mean_squared_error'
        )
        
        rmse_scores = np.sqrt(-cv_scores)
        results[name] = {
            'RMSE': rmse_scores.mean(),
            'Std Dev': rmse_scores.std()
        }
        
        # Fit the model on the full training data
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
    
    return results, fitted_models

# Function to make forecasts with fitted models
def make_forecasts(fitted_models, X_test):
    """
    Make forecasts using fitted models
    """
    forecasts = {}
    
    for name, model in fitted_models.items():
        predictions = model.predict(X_test)
        forecasts[name] = predictions
    
    return forecasts

# Function for SARIMAX time series model
def fit_sarimax_model(train_data, test_template, exog_cols=None):
    """
    Fit a SARIMAX model for time series forecasting
    """
    # Prepare data for SARIMAX
    train_data = train_data.sort_values('business_date')
    
    # Group by categorical variables if present
    results = {}
    
    if exog_cols and all(col in train_data.columns for col in exog_cols):
        # Group by the categorical columns
        groups = train_data.groupby(exog_cols)
        
        for group_name, group_data in groups:
            # Prepare time series data
            ts_data = group_data.set_index('business_date')['ledger_amount']
            
            # Create matching test data
            test_mask = True
            for i, col in enumerate(exog_cols):
                test_mask = test_mask & (test_template[col] == group_name[i] if isinstance(group_name, tuple) else test_template[col] == group_name)
            
            test_data = test_template[test_mask]
            
            if len(test_data) > 0 and len(ts_data) >= 2:  # Need at least 2 observations for SARIMAX
                try:
                    # Fit SARIMAX model
                    model = SARIMAX(ts_data, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4))
                    model_fit = model.fit(disp=False)
                    
                    # Forecast
                    forecast = model_fit.forecast(steps=1)
                    
                    # Store results
                    if isinstance(group_name, tuple):
                        group_key = "_".join(str(g) for g in group_name)
                    else:
                        group_key = str(group_name)
                        
                    results[group_key] = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
                except:
                    print(f"SARIMAX failed for group {group_name}. Using average.")
                    results[group_key] = ts_data.mean()
    else:
        # Fallback to simple time series on the entire dataset
        ts_data = train_data.set_index('business_date')['ledger_amount']
        
        try:
            model = SARIMAX(ts_data, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=1)
            results['overall'] = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
        except:
            print("SARIMAX failed. Using average.")
            results['overall'] = ts_data.mean()
    
    return results

# Function to analyze feature importance
def analyze_feature_importance(fitted_models, X_train, categorical_features):
    """
    Analyze and visualize feature importance
    """
    # Models that support feature importance
    models_with_importance = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
    
    for name in models_with_importance:
        if name in fitted_models:
            model = fitted_models[name]
            
            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            model_features = (
                preprocessor.transformers_[0][1].named_steps['imputer'].get_feature_names_out() +
                preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
            )
            
            # Get feature importances
            if name in ['Random Forest', 'Gradient Boosting']:
                importances = model.named_steps['model'].feature_importances_
            elif name == 'XGBoost':
                importances = model.named_steps['model'].feature_importances_
            elif name == 'LightGBM':
                importances = model.named_steps['model'].feature_importances_
            
            # Create DataFrame for visualization
            feature_importance_df = pd.DataFrame({
                'feature': model_features,
                'importance': importances
            })
            
            # Sort and plot
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Feature Importance - {name}')
            plt.tight_layout()
            plt.show()

# Main function to run the forecasting process
def forecast_ledger_amount(file_path, target_date='2023-09-30'):
    """
    Complete forecasting pipeline
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    print("Creating time features...")
    df_with_features = create_time_features(df)
    
    print("Preparing train/test sets...")
    # Identify categorical features
    categorical_features = ['currency', 'custom_1', 'custom_2', 'custom_3', 'account_description']
    categorical_features = [col for col in categorical_features if col in df_with_features.columns]
    
    # Prepare data
    X_train, y_train, X_test, test_template = prepare_train_test(df_with_features, target_date)
    
    print("Building and evaluating models...")
    results, fitted_models = build_and_evaluate_models(X_train, y_train, categorical_features)
    
    print("Model performance comparison:")
    for name, metrics in results.items():
        print(f"{name}: RMSE = {metrics['RMSE']:.2f} (±{metrics['Std Dev']:.2f})")
    
    print("\nMaking forecasts...")
    forecasts = make_forecasts(fitted_models, X_test)
    
    print("\nSARIMAX time series forecasts...")
    sarimax_forecasts = fit_sarimax_model(df_with_features, test_template, exog_cols=['account_description'])
    
    # Combine all forecasts into a DataFrame
    forecast_df = pd.DataFrame({name: pred for name, pred in forecasts.items()})
    
    # Add test template info
    for col in categorical_features:
        if col in test_template.columns:
            forecast_df[col] = test_template[col].values
    
    forecast_df['business_date'] = test_template['business_date'].values
    
    print("\nSummary of forecasts:")
    forecast_means = forecast_df.mean(axis=0)
    for model in forecasts.keys():
        print(f"{model}: {forecast_means[model]:.2f}")
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    
    for name, preds in forecasts.items():
        plt.plot(forecast_df.index, preds, label=name)
    
    plt.title("Forecasts by Different Models")
    plt.xlabel("Observation")
    plt.ylabel("Ledger Amount")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(fitted_models, X_train, categorical_features)
    
    return forecast_df, results, fitted_models

# Example usage
if __name__ == "__main__":
    # Replace with your file path
    file_path = "ledger_data.csv"
    
    # Target date for September 2023 (assuming quarterly data)
    target_date = '2023-09-30'
    
    # Run the forecasting process
    forecasts, results, models = forecast_ledger_amount(file_path, target_date)
    print("\nFinal forecasts:")
    print(forecasts)

# Sample function to generate synthetic data for testing
def generate_sample_data():
    """
    Generate sample quarterly data for testing
    """
    # Create date range for two quarters
    dates = [
        # March data
        *(['2023-03-01', '2023-03-15', '2023-03-31'] * 10),
        # June data
        *(['2023-06-01', '2023-06-15', '2023-06-30'] * 10)
    ]
    
    # Create currencies
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
    
    # Create other categorical features
    custom_1_values = ['Type1', 'Type2', 'Type3']
    custom_2_values = ['A', 'B', 'C', 'D']
    custom_3_values = ['High', 'Medium', 'Low']
    account_descriptions = ['Revenue', 'Expenses', 'Assets', 'Liabilities']
    
    # Generate sample data
    np.random.seed(42)
    n_samples = len(dates)
    
    data = {
        'business_date': dates,
        'currency': np.random.choice(currencies, n_samples),
        'custom_1': np.random.choice(custom_1_values, n_samples),
        'custom_2': np.random.choice(custom_2_values, n_samples),
        'custom_3': np.random.choice(custom_3_values, n_samples),
        'account_description': np.random.choice(account_descriptions, n_samples),
        'ledger_amount': np.random.normal(10000, 5000, n_samples)
    }
    
    # Create seasonal patterns and trends
    df = pd.DataFrame(data)
    df['business_date'] = pd.to_datetime(df['business_date'])
    
    # Add seasonality by account type
    for account in account_descriptions:
        mask = df['account_description'] == account
        
        # March amounts (Q1)
        march_mask = mask & (df['business_date'].dt.month == 3)
        
        # June amounts (Q2) - some increase for certain accounts
        june_mask = mask & (df['business_date'].dt.month == 6)
        
        if account == 'Revenue':
            df.loc[june_mask, 'ledger_amount'] = df.loc[june_mask, 'ledger_amount'] * 1.2  # 20% increase
        elif account == 'Expenses':
            df.loc[june_mask, 'ledger_amount'] = df.loc[june_mask, 'ledger_amount'] * 1.1  # 10% increase
    
    # Add some missing values
    mask = np.random.choice([True, False], n_samples, p=[0.05, 0.95])
    df.loc[mask, 'custom_2'] = np.nan
    
    # Save to CSV
    df.to_csv('sample_ledger_data.csv', index=False)
    print("Sample data generated and saved to 'sample_ledger_data.csv'")
    return df

# Uncomment to generate sample data
# sample_data = generate_sample_data()