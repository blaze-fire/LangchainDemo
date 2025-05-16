import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess data
def preprocess_data(file_path):
    """
    Load and preprocess the data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert business_date to datetime
    df['business_date'] = pd.to_datetime(df['business_date'])
    
    # Extract date features
    df['year'] = df['business_date'].dt.year
    df['month'] = df['business_date'].dt.month
    df['day'] = df['business_date'].dt.day
    df['quarter'] = df['business_date'].dt.quarter
    df['day_of_week'] = df['business_date'].dt.dayofweek
    df['day_of_year'] = df['business_date'].dt.dayofyear
    
    return df

# Function to handle categorical variables and missing values
def encode_categorical_features(df_train, df_test=None):
    """
    Encode categorical features and handle missing values
    
    Parameters:
    -----------
    df_train : DataFrame
        Training dataframe
    df_test : DataFrame or None
        Test dataframe. If None, only train data is processed
        
    Returns:
    --------
    df_train_encoded : DataFrame
        Encoded training dataframe
    df_test_encoded : DataFrame or None
        Encoded test dataframe (if df_test was provided)
    categorical_columns : list
        List of categorical column names
    encoders : dict
        Dictionary of encoders for each categorical column
    """
    # Identify categorical columns (excluding business_date which we'll handle separately)
    categorical_columns = ['currency', 'custom_1', 'custom_2', 'custom_3', 'account_description']
    
    # Check if all categorical columns exist in the dataframe
    categorical_columns = [col for col in categorical_columns if col in df_train.columns]
    
    # Create copies to avoid modifying the original dataframes
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy() if df_test is not None else None
    
    # Dictionary to store encoders
    encoders = {}
    
    # Process each categorical column
    for col in categorical_columns:
        # Impute missing values with 'Unknown'
        df_train_encoded[col] = df_train_encoded[col].fillna('Unknown')
        if df_test_encoded is not None:
            df_test_encoded[col] = df_test_encoded[col].fillna('Unknown')
        
        # Create and fit encoders
        encoder = LabelEncoder()
        df_train_encoded[col] = encoder.fit_transform(df_train_encoded[col])
        
        # Store encoder for later use
        encoders[col] = encoder
        
        # Transform test data if provided
        if df_test_encoded is not None:
            # Handle any new categories in test set that weren't in training
            df_test_encoded[col] = df_test_encoded[col].apply(
                lambda x: 'Unknown' if x not in encoder.classes_ else x
            )
            df_test_encoded[col] = encoder.transform(df_test_encoded[col])
    
    return df_train_encoded, df_test_encoded, categorical_columns, encoders

# Function to scale numerical features
def scale_numerical_features(df_train, df_test=None):
    """
    Scale numerical features
    """
    # Identify numerical columns
    numerical_columns = ['ledger_amount', 'year', 'month', 'day', 'quarter', 'day_of_week', 'day_of_year']
    numerical_columns = [col for col in numerical_columns if col in df_train.columns and col != 'ledger_amount']
    
    # Create copies
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy() if df_test is not None else None
    
    # Dictionary to store scalers
    scalers = {}
    
    # Scale numerical features
    for col in numerical_columns:
        scaler = MinMaxScaler()
        df_train_scaled[col] = scaler.fit_transform(df_train_scaled[[col]])
        
        # Store scaler
        scalers[col] = scaler
        
        # Scale test data if provided
        if df_test_scaled is not None:
            df_test_scaled[col] = scaler.transform(df_test_scaled[[col]])
    
    # Scale target separately
    target_scaler = MinMaxScaler()
    df_train_scaled['ledger_amount_scaled'] = target_scaler.fit_transform(df_train_scaled[['ledger_amount']])
    
    if df_test_scaled is not None:
        df_test_scaled['ledger_amount_scaled'] = target_scaler.transform(df_test_scaled[['ledger_amount']])
    
    scalers['ledger_amount'] = target_scaler
    
    return df_train_scaled, df_test_scaled, numerical_columns, scalers

# Function to prepare data for deep learning models (single input)
def prepare_model_inputs(df_train, df_test, categorical_columns, numerical_columns):
    """
    Prepare input features and targets for the model with a single combined input
    """
    # Features (X) and target (y) for training data
    feature_columns = categorical_columns + numerical_columns
    X_train = df_train[feature_columns].values
    y_train = df_train['ledger_amount_scaled'].values
    
    # Features (X) and target (y) for test data
    X_test = df_test[feature_columns].values
    y_test = df_test['ledger_amount_scaled'].values
    
    return X_train, y_train, X_test, y_test, feature_columns

# Build a basic DNN model with a single input
def build_dnn_model(input_dim):
    """
    Build a Deep Neural Network with a single input for all features
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    # Create sequential model
    model = Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # Hidden layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Build an LSTM model with a single input
def build_lstm_model(input_dim):
    """
    Build an LSTM model with a single input for all features
    """
    model = Sequential([
        # Input layer and reshape for LSTM
        layers.Input(shape=(input_dim,)),
        layers.Reshape((1, input_dim)),  # Reshape to (timesteps, features)
        
        # LSTM layers
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Build a Temporal Convolutional Network (TCN) with a single input
def build_tcn_model(input_dim):
    """
    Build a Temporal Convolutional Network with a single input for all features
    """
    model = Sequential([
        # Input layer and reshape for Conv1D
        layers.Input(shape=(input_dim,)),
        layers.Reshape((1, input_dim)),  # Reshape to (timesteps, features)
        
        # TCN-like architecture (Conv1D with dilations)
        layers.Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=1, activation='relu'),
        layers.Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=2, activation='relu'),
        layers.Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=4, activation='relu'),
        layers.GlobalMaxPooling1D(),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Define a Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, key_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(key_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build a Transformer model with a single input
def build_transformer_model(input_dim):
    """
    Build a Transformer model with a single input for all features
    """
    # Create Sequential model
    model = Sequential([
        # Input layer and reshape for Transformer
        layers.Input(shape=(input_dim,)),
        layers.Reshape((1, input_dim)),  # Reshape to (seq_length, features)
        
        # Add custom Transformer block
        # Note: We define this within the function to ensure compatibility with Sequential
        TransformerBlock(key_dim=input_dim, num_heads=2, ff_dim=32),
        
        # Pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test, target_scaler):
    """
    Evaluate model performance and return metrics
    """
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse scale predictions and true values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'y_true': y_true.flatten(),
        'y_pred': y_pred.flatten()
    }

# Plot actual vs predicted values
def plot_predictions(results, model_name):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results['y_true'], results['y_pred'], alpha=0.5)
    plt.plot([min(results['y_true']), max(results['y_true'])], 
             [min(results['y_true']), max(results['y_true'])], 'r--')
    plt.xlabel('Actual Ledger Amount')
    plt.ylabel('Predicted Ledger Amount')
    plt.title(f'{model_name} - Actual vs Predicted Ledger Amount')
    plt.text(0.05, 0.95, f"RMSE: {results['RMSE']:.2f}\nMAE: {results['MAE']:.2f}\nR²: {results['R2']:.2f}", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()

# Function to prepare data for the next quarter prediction
def prepare_next_quarter_data(df_train, next_quarter_month, encoders, scalers, categorical_columns, numerical_columns):
    """
    Prepare data for the next quarter prediction
    
    Parameters:
    -----------
    df_train : DataFrame
        Training dataframe
    next_quarter_month : int
        Month number of the next quarter (e.g., 9 for September)
    encoders : dict
        Dictionary of encoders for categorical columns
    scalers : dict
        Dictionary of scalers for numerical columns
    categorical_columns : list
        List of categorical column names
    numerical_columns : list
        List of numerical column names
        
    Returns:
    --------
    next_quarter_features : ndarray
        Features for the next quarter prediction
    """
    # Get unique combinations of categorical features from training data
    unique_cat_combinations = df_train[categorical_columns].drop_duplicates()
    
    # Create a dataframe for the next quarter
    next_quarter_df = pd.DataFrame()
    
    # Add categorical features
    for col in categorical_columns:
        next_quarter_df[col] = unique_cat_combinations[col]
    
    # Add the date features for the next quarter (September)
    # Assuming the year is the same as in the training data
    year = df_train['year'].max()  # Use the latest year from training data
    
    next_quarter_df['year'] = year
    next_quarter_df['month'] = next_quarter_month
    next_quarter_df['day'] = 1  # Default to first day of the month
    
    # Calculate other date features
    if next_quarter_month == 9:  # September
        next_quarter_df['quarter'] = 3
        # Approximate values for day_of_week and day_of_year
        next_quarter_df['day_of_week'] = 0  # Placeholder
        next_quarter_df['day_of_year'] = 244  # Approx. day of year for Sept 1
    
    # Scale numerical features
    for col in numerical_columns:
        if col in scalers:
            next_quarter_df[col] = scalers[col].transform(next_quarter_df[[col]])
    
    # Extract features for model input
    feature_columns = categorical_columns + numerical_columns
    X_next = next_quarter_df[feature_columns].values
    
    return X_next

# Main function to train models and predict
def train_and_predict(train_file_path, test_month=9):
    """
    Train models and predict ledger amounts for the next quarter
    
    Parameters:
    -----------
    train_file_path : str
        Path to the training data CSV file
    test_month : int
        Month number for the test quarter (e.g., 9 for September)
        
    Returns:
    --------
    predictions : dict
        Dictionary of predictions from each model
    """
    print("Loading and preprocessing data...")
    df = preprocess_data(train_file_path)
    
    # Create a validation set (80% train, 20% validation)
    train_months = [3, 6]  # March and June
    
    # Split data into train and validation
    mask = df['month'].isin(train_months)
    df_train = df[mask].copy()
    
    # Handle a small portion of data as validation
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Validation data shape: {df_val.shape}")
    
    # Encode categorical features
    df_train_encoded, df_val_encoded, categorical_columns, encoders = encode_categorical_features(df_train, df_val)
    
    # Scale numerical features
    df_train_scaled, df_val_scaled, numerical_columns, scalers = scale_numerical_features(df_train_encoded, df_val_encoded)
    
    # Prepare model inputs (with single combined input)
    X_train, y_train, X_val, y_val, feature_columns = prepare_model_inputs(
        df_train_scaled, df_val_scaled, categorical_columns, numerical_columns
    )
    
    # Calculate input dimension
    input_dim = X_train.shape[1]
    
    # Define models
    models = {
        'DNN': build_dnn_model(input_dim),
        'LSTM': build_lstm_model(input_dim),
        'TCN': build_tcn_model(input_dim),
        'Transformer': build_transformer_model(input_dim)
    }
    
    # Dictionary to store trained models and their results
    trained_models = {}
    results = {}
    
    # Training parameters
    epochs = 50
    batch_size = 32
    patience = 10
    
    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        print(f"\nEvaluating {name} model...")
        eval_results = evaluate_model(
            model, X_val, y_val, scalers['ledger_amount']
        )
        
        print(f"{name} Results:")
        print(f"  RMSE: {eval_results['RMSE']:.2f}")
        print(f"  MAE: {eval_results['MAE']:.2f}")
        print(f"  R²: {eval_results['R2']:.2f}")
        
        # Store model and results
        trained_models[name] = model
        results[name] = eval_results
        
        # Plot actual vs predicted
        plot_predictions(eval_results, name)
    
    # Prepare data for next quarter prediction
    print("\nPreparing data for next quarter prediction...")
    next_quarter_features = prepare_next_quarter_data(
        df_train, test_month, encoders, scalers, categorical_columns, numerical_columns
    )
    
    # Make predictions for next quarter
    predictions = {}
    for name, model in trained_models.items():
        print(f"Making predictions with {name} model...")
        pred_scaled = model.predict(next_quarter_features)
        pred = scalers['ledger_amount'].inverse_transform(pred_scaled)
        predictions[name] = pred
    
    return predictions, results, trained_models, df_train, categorical_columns, numerical_columns

# Example usage
if __name__ == "__main__":
    # Path to your data file
    data_file = "your_financial_data.csv"
    
    # September is month 9 (Q3)
    predictions, results, models, df_train, cat_cols, num_cols = train_and_predict(data_file, test_month=9)
    
    # Display predictions
    print("\nPredictions for September (Q3):")
    for model_name, pred in predictions.items():
        print(f"{model_name} Model:")
        print(f"  Mean Predicted Ledger Amount: {pred.mean():.2f}")
        print(f"  Min Predicted Ledger Amount: {pred.min():.2f}")
        print(f"  Max Predicted Ledger Amount: {pred.max():.2f}")
        
    # Find the best performing model
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
    print(f"\nBest performing model based on RMSE: {best_model}")
    
    # You can save the best model for future use
    models[best_model].save(f"best_model_{best_model}.h5")
    print(f"Best model saved as 'best_model_{best_model}.h5'")
