import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# Sample data loading (replace with your actual data)
def load_data():
    # For demonstration - you should replace this with your actual data loading
    # Create a sample dataset with 6 months of daily data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    np.random.seed(42)
    data = {
        'business_date': dates,
        'ledger_amount': np.random.normal(1000, 200, len(dates)) + np.arange(len(dates)) * 2,  # trend + noise
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], len(dates)),
        'custom_1': np.random.choice(['A', 'B', 'C'], len(dates)),
        'custom_2': np.random.choice(['X', 'Y', 'Z'], len(dates)),
        'custom_3': np.random.choice(['High', 'Medium', 'Low'], len(dates)),
        'account_description': np.random.choice(['Revenue', 'Expense', 'Asset', 'Liability'], len(dates))
    }
    
    df = pd.DataFrame(data)
    df['business_date'] = pd.to_datetime(df['business_date'])
    return df

# Data preprocessing
def preprocess_data(df):
    # Set business_date as index
    df = df.copy()
    df.set_index('business_date', inplace=True)
    
    # Extract date features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Identify categorical columns
    categorical_cols = ['currency', 'custom_1', 'custom_2', 'custom_3', 'account_description']
    numeric_cols = ['day_of_week', 'day_of_month', 'month', 'quarter']
    
    return df, categorical_cols, numeric_cols

# Feature Generation for Time Series
def create_lag_features(df, lag_days=[1, 7, 14]):
    df_lag = df.copy()
    for lag in lag_days:
        df_lag[f'ledger_amount_lag_{lag}'] = df_lag['ledger_amount'].shift(lag)
    
    # Create rolling window features
    df_lag['ledger_amount_rolling_mean_7'] = df_lag['ledger_amount'].rolling(window=7).mean().shift(1)
    df_lag['ledger_amount_rolling_mean_14'] = df_lag['ledger_amount'].rolling(window=14).mean().shift(1)
    df_lag['ledger_amount_rolling_std_7'] = df_lag['ledger_amount'].rolling(window=7).std().shift(1)
    
    # Drop NaN values created by lag features
    df_lag = df_lag.dropna()
    
    return df_lag

# Split data for time series evaluation
def split_time_series_data(df):
    # Use 80% of data for training, keep last 20% for validation
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:]
    
    return train_data, val_data

# Prepare data for ML models
def prepare_ml_features(df, categorical_cols, numeric_cols):
    X = df.drop('ledger_amount', axis=1)
    y = df['ledger_amount']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
    
    return X, y, preprocessor

# Create data for sequence models
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to generate future dates
def generate_future_dates(last_date, periods=31):
    date_range = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    return date_range

#################################################
# 1. STATISTICAL MODELS
#################################################

# ARIMA/SARIMA model
def run_sarima(df):
    print("Running SARIMA model...")
    # Prepare the data
    ts_data = df['ledger_amount']
    
    # Split data
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    # Fit SARIMA model - example parameters (should be tuned)
    model = SARIMAX(train_data, 
                    order=(1, 1, 1),  # (p, d, q)
                    seasonal_order=(1, 1, 1, 7),  # (P, D, Q, S)
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    
    # Make predictions for the test set
    predictions = model_fit.forecast(steps=len(test_data))
    
    # Evaluate the model
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    print(f"SARIMA - Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Forecast for July (assuming July has 31 days)
    last_date = df.index[-1]
    forecast_dates = generate_future_dates(last_date, 31)
    forecast = model_fit.forecast(steps=31)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'business_date': forecast_dates, 'predicted_ledger_amount': forecast})
    forecast_df.set_index('business_date', inplace=True)
    
    return forecast_df, model_fit

# Exponential Smoothing
def run_exponential_smoothing(df):
    print("Running Exponential Smoothing model...")
    # Prepare the data
    ts_data = df['ledger_amount']
    
    # Split data
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    # Fit Exponential Smoothing model
    model = ExponentialSmoothing(train_data, 
                                trend='add',
                                seasonal='add', 
                                seasonal_periods=7)  # assuming weekly seasonality
    
    model_fit = model.fit()
    
    # Make predictions for the test set
    predictions = model_fit.forecast(len(test_data))
    
    # Evaluate the model
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    print(f"ExponentialSmoothing - Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Forecast for July (assuming July has 31 days)
    last_date = df.index[-1]
    forecast_dates = generate_future_dates(last_date, 31)
    forecast = model_fit.forecast(31)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'business_date': forecast_dates, 'predicted_ledger_amount': forecast})
    forecast_df.set_index('business_date', inplace=True)
    
    return forecast_df, model_fit

# Prophet model
def run_prophet(df):
    print("Running Prophet model...")
    # Prepare data for Prophet
    prophet_df = df.reset_index()[['business_date', 'ledger_amount']].rename(columns={
        'business_date': 'ds', 
        'ledger_amount': 'y'
    })
    
    # Split data
    train_size = int(len(prophet_df) * 0.8)
    train_data = prophet_df[:train_size]
    test_data = prophet_df[train_size:]
    
    # Fit Prophet model
    model = Prophet(
        yearly_seasonality=False,  # Only 6 months of data
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    
    # Add categorical regressors if needed
    # For demonstration, we're not using them, but you could add:
    # df_with_cat = df.reset_index()
    # for cat_col in categorical_cols:
    #     prophet_df[cat_col] = df_with_cat[cat_col]
    #     model.add_regressor(cat_col)
    
    model.fit(train_data)
    
    # Create forecast dataframe for test period
    test_dates = prophet_df.iloc[train_size:]['ds']
    future_test = pd.DataFrame({'ds': test_dates})
    forecast_test = model.predict(future_test)
    
    # Evaluate on test set
    mae = mean_absolute_error(test_data['y'], forecast_test['yhat'])
    rmse = np.sqrt(mean_squared_error(test_data['y'], forecast_test['yhat']))
    print(f"Prophet - Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Forecast for July
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, 31)
    future = pd.DataFrame({'ds': future_dates})
    
    # Make the forecast
    forecast = model.predict(future)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'business_date': forecast['ds'],
        'predicted_ledger_amount': forecast['yhat']
    })
    forecast_df.set_index('business_date', inplace=True)
    
    return forecast_df, model

#################################################
# 2. MACHINE LEARNING MODELS
#################################################

# Linear Regression
def run_linear_regression(df, categorical_cols, numeric_cols):
    print("Running Linear Regression model...")
    # Create lag features
    df_with_features = create_lag_features(df)
    
    # Prepare features and target
    X, y, preprocessor = prepare_ml_features(df_with_features, categorical_cols, numeric_cols)
    
    # Split data
    train_data, val_data = split_time_series_data(df_with_features)
    X_train, y_train = prepare_ml_features(train_data, categorical_cols, numeric_cols)[:2]
    X_val, y_val = prepare_ml_features(val_data, categorical_cols, numeric_cols)[:2]
    
    # Create and train pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    print(f"Linear Regression - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Now let's prepare data for July forecast
    # We need to create a feature set that extends into July
    # This is a simplified version - in reality, you'd need to incrementally
    # forecast each day and add it back to the feature set
    
    # For demonstration, let's assume we're forecasting just the first week of July
    # using the trained model and the last available features
    last_features = X.iloc[-1:].copy()  # Get the last observation's features
    
    # Generate forecast for July
    # In a real application, you'd need to:
    # 1. Incrementally forecast each day
    # 2. Update features (including lags) with each forecast
    # 3. Then predict the next day
    # This is simplified for demonstration
    
    return model, preprocessor

# Random Forest
def run_random_forest(df, categorical_cols, numeric_cols):
    print("Running Random Forest model...")
    # Create lag features
    df_with_features = create_lag_features(df)
    
    # Prepare features and target
    X, y, preprocessor = prepare_ml_features(df_with_features, categorical_cols, numeric_cols)
    
    # Split data
    train_data, val_data = split_time_series_data(df_with_features)
    X_train, y_train = prepare_ml_features(train_data, categorical_cols, numeric_cols)[:2]
    X_val, y_val = prepare_ml_features(val_data, categorical_cols, numeric_cols)[:2]
    
    # Create and train pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    print(f"Random Forest - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return model, preprocessor

# XGBoost
def run_xgboost(df, categorical_cols, numeric_cols):
    print("Running XGBoost model...")
    # Create lag features
    df_with_features = create_lag_features(df)
    
    # Prepare features and target
    X, y, preprocessor = prepare_ml_features(df_with_features, categorical_cols, numeric_cols)
    
    # Split data
    train_data, val_data = split_time_series_data(df_with_features)
    X_train, y_train = prepare_ml_features(train_data, categorical_cols, numeric_cols)[:2]
    X_val, y_val = prepare_ml_features(val_data, categorical_cols, numeric_cols)[:2]
    
    # Create and train pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbosity=0
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    print(f"XGBoost - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return model, preprocessor

# LightGBM
def run_lightgbm(df, categorical_cols, numeric_cols):
    print("Running LightGBM model...")
    # Create lag features
    df_with_features = create_lag_features(df)
    
    # Prepare features and target
    X, y, preprocessor = prepare_ml_features(df_with_features, categorical_cols, numeric_cols)
    
    # Split data
    train_data, val_data = split_time_series_data(df_with_features)
    X_train, y_train = prepare_ml_features(train_data, categorical_cols, numeric_cols)[:2]
    X_val, y_val = prepare_ml_features(val_data, categorical_cols, numeric_cols)[:2]
    
    # Create and train pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    print(f"LightGBM - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return model, preprocessor

#################################################
# 3. DEEP LEARNING MODELS
#################################################

# Create features for deep learning models
def prepare_dl_features(df, categorical_cols, look_back=30):
    # Normalize continuous variables
    scaler = StandardScaler()
    scaled_amount = scaler.fit_transform(df[['ledger_amount']])
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse=False)
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    
    # Combine features
    features = np.hstack((scaled_amount, encoded_cats))
    
    # Create sequences
    X, y = create_sequences(features, look_back)
    
    # The target is the ledger amount (first column in our features)
    y = y[:, 0]
    
    return X, y, scaler, encoder

# LSTM Network
def run_lstm(df, categorical_cols):
    print("Running LSTM model...")
    # Prepare data
    look_back = 30  # Use 30 days of history to predict next day
    X, y, scaler, encoder = prepare_dl_features(df, categorical_cols, look_back)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(25, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    
    # Inverse transform to get actual values
    y_val_original = y_val.reshape(-1, 1)  # Reshape for inverse_transform
    y_pred_original = y_pred
    
    # Calculate metrics on original scale
    mae = mean_absolute_error(y_val_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
    print(f"LSTM - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return model

# GRU Network
def run_gru(df, categorical_cols):
    print("Running GRU model...")
    # Prepare data
    look_back = 30  # Use 30 days of history to predict next day
    X, y, scaler, encoder = prepare_dl_features(df, categorical_cols, look_back)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Define GRU model
    model = Sequential([
        GRU(50, activation='relu', input_shape=(look_back, X.shape[2]), return_sequences=True),
        Dropout(0.2),
        GRU(25, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"GRU - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return model

# Temporal Convolutional Network (TCN)
def run_tcn(df, categorical_cols):
    print("Running TCN model...")
    # Prepare data
    look_back = 30  # Use 30 days of history to predict next day
    X, y, scaler, encoder = prepare_dl_features(df, categorical_cols, look_back)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Define TCN model (using 1D convolutions)
    model = Sequential([
        # First convolutional layer
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X.shape[2])),
        MaxPooling1D(pool_size=2),
        
        # Second convolutional layer
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten layer
        Flatten(),
        
        # Dense layers
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"TCN - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return model

# Transformer-based model
def run_transformer(df, categorical_cols):
    print("Running Transformer model...")
    # Prepare data
    look_back = 30  # Use 30 days of history to predict next day
    X, y, scaler, encoder = prepare_dl_features(df, categorical_cols, look_back)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Define Transformer model
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(attention_output)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return ffn_output
    
    # Build the model
    inputs = tf.keras.Input(shape=(look_back, X.shape[2]))
    x = inputs
    
    # Transformer layers
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
    
    # Output layer
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Transformer - Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return model

#################################################
# FORECASTING FOR JULY
#################################################

def generate_july_forecasts(df, categorical_cols, numeric_cols):
    """Generate forecasts for July using the best models"""
    # We'll assume July has 31 days
    last_date = df.index[-1]  # Last date in our dataset (end of June)
    july_dates = generate_future_dates(last_date, 31)
    
    forecasts = {}
    
    # 1. SARIMA forecast
    sarima_forecast, _ = run_sarima(df)
    forecasts['SARIMA'] = sarima_forecast
    
    # 2. Exponential Smoothing forecast
    es_forecast, _ = run_exponential_smoothing(df)
    forecasts['ExponentialSmoothing'] = es_forecast
    
    # 3. Prophet forecast
    prophet_forecast, _ = run_prophet(df)
    forecasts['Prophet'] = prophet_forecast
    
    # 4. For ML models, we need a different approach due to lag features
    # This is a simplified example - in a real implementation, you would:
    # - Train the ML model on the entire dataset
    # - For each day in July:
    #   - Create features including lags
    #   - Make a prediction
    #   - Add the prediction to your dataset
    #   - Move to the next day
    
    # 5. For DL models, similar to ML models, you need to:
    # - Generate sequences for prediction
    # - Make predictions for each day
    # - Update sequences for the next day's prediction
    
    # For demonstration, we'll just show a simple forecast visualization
    plt.figure(figsize=(12, 6))
    
    # Plot the historical data
    plt.plot(df.index, df['ledger_amount'], label='Historical Data')
    
    # Plot forecasts
    for model_name, forecast_df in forecasts.items():
        plt.plot(forecast_df.index, forecast_df['