import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from tensorflow.keras.models import Sequential
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Improved LSTM Model with regularization
def build_improved_lstm(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build an improved LSTM model with proper regularization and architecture
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # LSTM Layer with regularization
        layers.LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=False
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 2. Bidirectional LSTM for capturing forward and backward patterns
def build_bilstm_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a Bidirectional LSTM model that processes sequences in both directions
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled BiLSTM model
    """
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Bidirectional LSTM Layer
        layers.Bidirectional(
            layers.LSTM(
                units=units,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2,
                return_sequences=False
            )
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 3. Stacked LSTM with multiple LSTM layers
def build_stacked_lstm_model(input_shape, units=[64, 32], dropout_rate=0.3, learning_rate=0.001):
    """
    Build a stacked LSTM model with multiple LSTM layers
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : list
        List of LSTM units for each layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled stacked LSTM model
    """
    model = Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Add multiple LSTM layers
    for i, unit in enumerate(units):
        return_sequences = i < len(units) - 1  # Return sequences for all but last LSTM layer
        
        model.add(layers.LSTM(
            units=unit,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=return_sequences
        ))
        
        if return_sequences:
            model.add(layers.BatchNormalization())
    
    # Dense layers for prediction
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(dropout_rate/2))
    
    # Output layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 4. Stacked Bidirectional LSTM
def build_stacked_bilstm_model(input_shape, units=[64, 32], dropout_rate=0.3, learning_rate=0.001):
    """
    Build a stacked bidirectional LSTM model
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : list
        List of LSTM units for each layer
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled stacked BiLSTM model
    """
    model = Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Add multiple BiLSTM layers
    for i, unit in enumerate(units):
        return_sequences = i < len(units) - 1  # Return sequences for all but last LSTM layer
        
        model.add(layers.Bidirectional(
            layers.LSTM(
                units=unit,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2,
                return_sequences=return_sequences
            )
        ))
        
        if return_sequences:
            model.add(layers.BatchNormalization())
    
    # Dense layers for prediction
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(dropout_rate/2))
    
    # Output layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 5. Convolutional LSTM - combines CNN and LSTM for better feature extraction
def build_conv_lstm_model(input_shape, filters=64, lstm_units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a CNN-LSTM model that uses convolutional layers before LSTM
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    filters : int
        Number of CNN filters
    lstm_units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled CNN-LSTM model
    """
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Reshape for Conv1D
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # CNN layers for feature extraction
        layers.Conv2D(
            filters=filters, 
            kernel_size=(3, 1), 
            activation='relu', 
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        
        # Reshape back for LSTM
        layers.Reshape((input_shape[0], filters)),
        
        # LSTM layer
        layers.LSTM(
            units=lstm_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=False
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 6. LSTM with Attention Mechanism
def build_lstm_with_attention(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build an LSTM model with attention mechanism
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled LSTM model with attention
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layer with return sequences
    lstm_out = layers.LSTM(
        units=units,
        activation='tanh',
        recurrent_activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        return_sequences=True
    )(inputs)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    
    # Apply attention weights
    context = layers.Dot(axes=1)([lstm_out, layers.Reshape((input_shape[0], 1))(attention_weights)])
    context = layers.Flatten()(context)
    
    # Dense layers for prediction
    x = layers.BatchNormalization()(context)
    x = layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(dropout_rate/2)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='linear')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 7. BiLSTM with Attention Mechanism
def build_bilstm_with_attention(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a BiLSTM model with attention mechanism
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled BiLSTM model with attention
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # BiLSTM layer with return sequences
    bilstm_out = layers.Bidirectional(
        layers.LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=True
        )
    )(inputs)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(bilstm_out)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    
    # Apply attention weights
    context = layers.Dot(axes=1)([bilstm_out, layers.Reshape((input_shape[0], 1))(attention_weights)])
    context = layers.Flatten()(context)
    
    # Dense layers for prediction
    x = layers.BatchNormalization()(context)
    x = layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(dropout_rate/2)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='linear')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 8. GRU (Gated Recurrent Unit) - Alternative to LSTM
def build_gru_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a GRU model as an alternative to LSTM
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of GRU units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled GRU model
    """
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # GRU Layer with regularization
        layers.GRU(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=False
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 9. Bidirectional GRU
def build_bigru_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a Bidirectional GRU model
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of GRU units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled BiGRU model
    """
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Bidirectional GRU Layer
        layers.Bidirectional(
            layers.GRU(
                units=units,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2,
                return_sequences=False
            )
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# 10. TimeDistributed LSTM model for advanced feature extraction
def build_time_distributed_lstm_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a TimeDistributed LSTM model for advanced feature extraction
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled TimeDistributed LSTM model
    """
    # Reshape input to have an extra dimension for TimeDistributed
    # The input shape will be (batch_size, timesteps, 1, features)
    reshaped_input_shape = (input_shape[0], 1, input_shape[1])
    
    model = Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Reshape for TimeDistributed
        layers.Reshape((input_shape[0], 1, input_shape[1])),
        
        # TimeDistributed Dense for feature extraction at each timestep
        layers.TimeDistributed(
            layers.Dense(32, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001))
        ),
        
        # Reshape back for LSTM
        layers.Reshape((input_shape[0], 32)),
        
        # LSTM Layer
        layers.LSTM(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            return_sequences=False
        ),
        
        # Dense layers for prediction
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use AMSGrad variant of Adam which has better convergence properties
    optimizer = optimizers.Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=True
    )
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

# Helper function to find optimal learning rate
def build_with_lr_finder(model_builder, input_shape, **kwargs):
    """
    Build model with LR finder callback to determine optimal learning rate
    
    Parameters:
    -----------
    model_builder : function
        Model building function
    input_shape : tuple
        Shape of input data
    **kwargs : dict
        Additional arguments for model_builder
    
    Returns:
    --------
    model : tf.keras.Model
        Model with LR finder
    lr_callback : tf.keras.callbacks.LearningRateScheduler
        LR finder callback
    """
    # Create model with initial low learning rate
    model = model_builder(input_shape, learning_rate=1e-6, **kwargs)
    
    # Create learning rate finder callback
    class LRFinder(callbacks.Callback):
        def __init__(self, min_lr=1e-6, max_lr=1e-1, n_steps=100):
            super(LRFinder, self).__init__()
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.n_steps = n_steps
            self.lrs = []
            self.losses = []
            
        def on_train_begin(self, logs=None):
            self.lrs = []
            self.losses = []
            
        def on_batch_end(self, batch, logs=None):
            # Calculate current learning rate
            step = self.params['steps'] * self.params['epochs'] * batch / self.n_steps
            lr = self.min_lr * (self.max_lr / self.min_lr) ** min(1, step / self.n_steps)
            
            # Set learning rate
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
            # Store learning rate and loss
            self.lrs.append(lr)
            self.losses.append(logs['loss'])
            
        def plot_lr_finder(self):
            """Plot learning rate finder results"""
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.semilogx(self.lrs, self.losses)
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True)
            plt.show()
    
    lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, n_steps=100)
    
    return model, lr_finder

# Function to create an LSTM ensemble model
def create_lstm_ensemble(input_shape, n_models=3):
    """
    Create an ensemble of LSTM models for better performance
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data in format (timesteps, features)
    n_models : int
        Number of models in the ensemble
    
    Returns:
    --------
    ensemble : list
        List of compiled LSTM models
    """
    ensemble = []
    
    # Create different types of models for the ensemble
    model_types = [
        build_improved_lstm,
        build_bilstm_model,
        build_stacked_lstm_model,
        build_stacked_bilstm_model,
        build_lstm_with_attention,
        build_bilstm_with_attention,
        build_gru_model,
        build_bigru_model
    ]
    
    # Select n_models model types
    selected_model_types = np.random.choice(model_types, min(n_models, len(model_types)), replace=False)
    
    # Create models
    for i, model_builder in enumerate(selected_model_types):
        if model_builder == build_stacked_lstm_model or model_builder == build_stacked_bilstm_model:
            model = model_builder(
                input_shape=input_shape,
                units=[64, 32],
                dropout_rate=0.2 + 0.1*i,  # Vary dropout for diversity
                learning_rate=0.001 * (0.8 + 0.4*i)  # Vary learning rate for diversity
            )
        else:
            model = model_builder(
                input_shape=input_shape,
                units=64,
                dropout_rate=0.2 + 0.1*i,  # Vary dropout for diversity
                learning_rate=0.001 * (0.8 + 0.4*i)  # Vary learning rate for diversity
            )
        
        ensemble.append(model)
    
    return ensemble

# Function to predict with ensemble
def predict_with_lstm_ensemble(ensemble, X):
    """
    Make predictions using an ensemble of LSTM models
    
    Parameters:
    -----------
    ensemble : list
        List of trained LSTM models
    X : numpy.ndarray
        Input data
    
    Returns:
    --------
    predictions : numpy.ndarray
        Ensemble predictions
    """
    # Get predictions from all models
    all_preds = []
    
    for model in ensemble:
        preds = model.predict(X)
        all_preds.append(preds)
    
    # Average predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    
    return ensemble_preds