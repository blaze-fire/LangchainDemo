import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
import itertools
import time
import json
from datetime import datetime
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer for sequence models
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        # score shape: (batch_size, seq_len, 1)
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch_size, features)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector, attention_weights
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

class TextNumericClassifier:
    def __init__(self, max_words=10000, max_sequence_length=100, embedding_dim=100, l2_reg=0.01):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.l2_reg = l2_reg
        self.model = None
        self.tokenizer = None
        self.threshold = 0.5
        
    def create_model(self):
        # Get actual vocabulary size
        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_words)
        
        # Text input branch
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        embedding = Embedding(vocab_size, self.embedding_dim)(text_input)
        
        # Bidirectional LSTM layer
        bilstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embedding)
        
        # Apply attention mechanism
        context_vector, attention_weights = AttentionLayer()(bilstm)
        
        # Numeric input branch
        numeric_input = Input(shape=(1,), name='numeric_input')
        
        # Combine text and numeric features
        combined = Concatenate()([context_vector, numeric_input])
        
        # Dense layers with configurable regularization
        dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(combined)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(dropout1)
        dropout2 = Dropout(0.5)(dense2)
        
        # Output layer with sigmoid activation for binary classification
        output = Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(inputs=[text_input, numeric_input], outputs=output)
        
        # Custom metrics to track recall
        recall = tf.keras.metrics.Recall(name='recall')
        precision = tf.keras.metrics.Precision(name='precision')
        
        # Compile model with default loss initially
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', precision, recall]
        )
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, use_smote=True, pos_weight=15.0, 
            epochs=30, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the model with options for handling class imbalance
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE for oversampling
            pos_weight: Positive class weight for weighted loss
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Process text data if it's the first time fitting
        if self.tokenizer is None:
            # Extract text and numeric features
            texts = [item[0] for item in X_train]
            numeric_values = np.array([float(item[1]) for item in X_train]).reshape(-1, 1)
            
            # Tokenize text
            self.tokenizer = Tokenizer(num_words=self.max_words)
            self.tokenizer.fit_on_texts(texts)
            
            # Create and compile model
            self.create_model()
        else:
            # If already fitted, just extract features
            texts = [item[0] for item in X_train]
            numeric_values = np.array([float(item[1]) for item in X_train]).reshape(-1, 1)
        
        # Convert text to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Apply SMOTE if requested
        if use_smote:
            from imblearn.over_sampling import SMOTE
            # Reshape for SMOTE
            flat_sequences = padded_sequences.reshape(padded_sequences.shape[0], -1)
            combined_features = np.hstack((flat_sequences, numeric_values))
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            resampled_features, resampled_labels = smote.fit_resample(combined_features, y_train)
            
            # Split back into sequence and numeric
            padded_sequences = resampled_features[:, :-1].reshape(-1, self.max_sequence_length)
            numeric_values = resampled_features[:, -1].reshape(-1, 1)
            y_train = resampled_labels
        
        # Use class_weights to handle imbalance if not using SMOTE
        if not use_smote:
            class_weights = {0: 1.0, 1: pos_weight}
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )
            
            history = self.model.fit(
                [padded_sequences, numeric_values], 
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                class_weight=class_weights
            )
        else:
            # Train without class weights if using SMOTE
            history = self.model.fit(
                [padded_sequences, numeric_values], 
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose
            )
        
        # Adjust threshold to optimize recall if needed
        if validation_split > 0:
            # Use a portion of training data as validation
            val_size = int(len(padded_sequences) * validation_split)
            val_indices = np.arange(len(padded_sequences) - val_size, len(padded_sequences))
            
            val_sequences = padded_sequences[val_indices]
            val_numeric = numeric_values[val_indices]
            val_labels = y_train[val_indices]
            
            # Get predictions
            val_preds = self.model.predict([val_sequences, val_numeric])
            
            # Find threshold that gives desired recall
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                threshold_preds = (val_preds > threshold).astype(int)
                
                # Calculate metrics
                r = recall_score(val_labels, threshold_preds)
                p = precision_score(val_labels, threshold_preds, zero_division=0)
                f1 = f1_score(val_labels, threshold_preds, zero_division=0)
                
                # Update if this is better
                if r >= 0.95 and f1 > best_f1:  # Target at least 95% recall
                    best_f1 = f1
                    best_threshold = threshold
            
            self.threshold = best_threshold
            print(f"Adjusted decision threshold to {self.threshold:.2f} for optimal recall-precision balance")
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions using the trained model
        
        Args:
            X_test: Test features
            
        Returns:
            Binary predictions
        """
        # Extract text and numeric features
        texts = [item[0] for item in X_test]
        numeric_values = np.array([float(item[1]) for item in X_test]).reshape(-1, 1)
        
        # Convert text to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Make predictions
        y_pred_proba = self.model.predict([padded_sequences, numeric_values])
        
        # Apply threshold
        y_pred = (y_pred_proba > self.threshold).astype(int).flatten()
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "threshold": self.threshold
        }
    
    def save_model(self, filepath):
        """
        Save the model and tokenizer
        
        Args:
            filepath: Base path to save the model
        """
        # Save Keras model
        self.model.save(f"{filepath}_model.h5")
        
        # Save tokenizer
        with open(f"{filepath}_tokenizer.json", 'w') as f:
            json.dump(self.tokenizer.to_json(), f)
        
        # Save configuration
        config = {
            "max_words": self.max_words,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dim": self.embedding_dim,
            "l2_reg": self.l2_reg,
            "threshold": self.threshold
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f)
            
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath: Base path to load the model from
        """
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        
        # Load config
        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)
            
        self.max_words = config["max_words"]
        self.max_sequence_length = config["max_sequence_length"]
        self.embedding_dim = config["embedding_dim"]
        self.l2_reg = config["l2_reg"]
        self.threshold = config["threshold"]
        
        # Load tokenizer
        with open(f"{filepath}_tokenizer.json", 'r') as f:
            tokenizer_json = json.load(f)
            self.tokenizer = tokenizer_from_json(tokenizer_json)
        
        # Load model with custom AttentionLayer
        self.model = load_model(
            f"{filepath}_model.h5",
            custom_objects={"AttentionLayer": AttentionLayer}
        )

def evaluate_params(params, X, y, n_splits=5):
    """
    Evaluate a single parameter combination using cross-validation
    
    Args:
        params: Dictionary of parameters to use
        X: Features
        y: Labels
        n_splits: Number of cross-validation folds
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating: {params}")
    start_time = time.time()
    
    # Use stratified k-fold to maintain class distribution
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Create and train model
        model = TextNumericClassifier(
            max_words=params.get('max_words', 10000),
            max_sequence_length=params.get('max_sequence_length', 100),
            embedding_dim=params.get('embedding_dim', 100),
            l2_reg=params.get('l2_reg', 0.01)
        )
        
        # Train on this fold
        model.fit(
            X_fold_train, 
            y_fold_train,
            use_smote=params.get('use_smote', True),
            pos_weight=params.get('pos_weight', 15.0),
            epochs=params.get('epochs', 30),
            batch_size=params.get('batch_size', 32),
            validation_split=0.1,  # Small validation split within the training fold
            verbose=0
        )
        
        # Evaluate on validation fold
        y_pred = model.predict(X_fold_val)
        fold_recall = recall_score(y_fold_val, y_pred)
        fold_precision = precision_score(y_fold_val, y_pred, zero_division=0)
        fold_f1 = f1_score(y_fold_val, y_pred, zero_division=0)
        
        fold_scores.append({
            'fold': fold,
            'recall': fold_recall,
            'precision': fold_precision,
            'f1': fold_f1,
            'threshold': model.threshold
        })
        
        print(f"  Fold {fold+1}: Recall={fold_recall:.4f}, Precision={fold_precision:.4f}, F1={fold_f1:.4f}")
    
    # Calculate average metrics across folds
    avg_recall = np.mean([s['recall'] for s in fold_scores])
    avg_precision = np.mean([s['precision'] for s in fold_scores])
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    avg_threshold = np.mean([s['threshold'] for s in fold_scores])
    
    # Calculate standard deviation for stability assessment
    std_recall = np.std([s['recall'] for s in fold_scores])
    
    # Record results
    duration = time.time() - start_time
    result = {
        **params,
        'avg_recall': avg_recall,
        'avg_precision': avg_precision,
        'avg_f1': avg_f1,
        'avg_threshold': avg_threshold,
        'std_recall': std_recall,  # Stability metric
        'fold_details': fold_scores,
        'duration_seconds': duration
    }
    
    print(f"Average: Recall={avg_recall:.4f}, Precision={avg_precision:.4f}, F1={avg_f1:.4f}, Time={duration:.2f}s")
    return result

def tune_hyperparameters(X, y, param_grid, n_splits=5, n_jobs=-1, target_recall=0.99):
    """
    Perform hyperparameter tuning with parallel cross-validation
    
    Args:
        X: Features
        y: Labels
        param_grid: Dictionary of parameters to explore
        n_splits: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 for all available cores)
        target_recall: Minimum recall to consider for best model selection
    
    Returns:
        best_params: Dictionary of best parameters
        results_df: DataFrame with all results
    """
    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [dict(zip(param_keys, values)) 
                          for values in itertools.product(*param_values)]
    
    print(f"Tuning with {len(param_combinations)} parameter combinations using {n_jobs} parallel jobs")
    
    # Run evaluations in parallel
    timestamp_start = datetime.now().strftime("%Y%m%d_%H%M")
    try:
        # Process each parameter combination in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_params)(params, X, y, n_splits)
            for params in param_combinations
        )
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        # Fallback to sequential processing
        print("Falling back to sequential processing...")
        results = []
        for params in param_combinations:
            try:
                result = evaluate_params(params, X, y, n_splits)
                results.append(result)
                
                # Save interim results after each combination in case of interruption
                interim_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'fold_details'} 
                                          for r in results])
                interim_df.to_csv(f'interim_tuning_results_{timestamp_start}.csv', index=False)
            except Exception as e:
                print(f"Error evaluating {params}: {e}")
    
    # Prepare results DataFrame (exclude detailed fold information for clarity)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'fold_details'} 
                              for r in results])
    
    # Find best parameters based on target recall and highest precision
    top_recall_mask = results_df['avg_recall'] >= target_recall
    if top_recall_mask.sum() > 0:
        # First priority: Among models with target recall, find highest precision
        top_recall_df = results_df[top_recall_mask]
        best_idx = top_recall_df['avg_precision'].idxmax()
    else:
        # If no model meets target recall, just get highest recall
        best_idx = results_df['avg_recall'].idxmax()
    
    best_params = {k: results_df.loc[best_idx][k] for k in param_keys}
    best_metrics = {
        'recall': results_df.loc[best_idx]['avg_recall'],
        'precision': results_df.loc[best_idx]['avg_precision'],
        'f1': results_df.loc[best_idx]['avg_f1'],
        'threshold': results_df.loc[best_idx]['avg_threshold']
    }
    
    print("\nBest parameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print("\nBest metrics:")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_df.to_csv(f'tuning_results_{timestamp}.csv', index=False)
    
    # Save best parameters
    with open(f'best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save detailed results including fold information
    with open(f'detailed_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_params, results_df, best_metrics

def phased_tuning(X, y, n_splits=5, n_jobs=-1):
    """
    Implement phased hyperparameter tuning to make the process more efficient
    
    Args:
        X: Features
        y: Labels
        n_splits: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        
    Returns:
        Final best parameters
    """
    print("Starting phased hyperparameter tuning")
    
    # Phase 1: Focus on most critical parameters for recall
    print("\n--- Phase 1: Optimizing for recall-critical parameters ---")
    phase1_param_grid = {
        'pos_weight': [10.0, 15.0, 20.0],      # Most critical for recall
        'max_words': [10000],                   # Keep fixed initially
        'embedding_dim': [100],                 # Keep fixed initially
        'l2_reg': [0.01],                       # Keep fixed initially
        'batch_size': [32],                     # Keep fixed initially
        'use_smote': [True]                     # Keep fixed initially
    }
    
    best_params_phase1, _, _ = tune_hyperparameters(
        X, y, phase1_param_grid, n_splits=n_splits, n_jobs=n_jobs
    )
    
    # Phase 2: Expand to include other parameters around the best pos_weight
    print("\n--- Phase 2: Fine-tuning model architecture ---")
    phase2_param_grid = {
        'pos_weight': [best_params_phase1['pos_weight']],  # Fixed at best value
        'max_words': [5000, 10000, 15000],      # Now explore vocabulary size
        'embedding_dim': [50, 100, 200],        # Now explore embedding dimensions
        'l2_reg': [0.001, 0.01, 0.1],           # Now explore regularization
        'batch_size': [32],                     # Keep fixed
        'use_smote': [True]                     # Keep fixed
    }
    
    best_params_phase2, _, _ = tune_hyperparameters(
        X, y, phase2_param_grid, n_splits=n_splits, n_jobs=n_jobs
    )
    
    # Phase 3: Fine-tune training parameters with best architecture
    print("\n--- Phase 3: Fine-tuning training parameters ---")
    phase3_param_grid = {
        'pos_weight': [best_params_phase1['pos_weight']],      # Fixed at best value
        'max_words': [best_params_phase2['max_words']],        # Fixed at best value
        'embedding_dim': [best_params_phase2['embedding_dim']], # Fixed at best value
        'l2_reg': [best_params_phase2['l2_reg']],              # Fixed at best value
        'batch_size': [16, 32, 64],                            # Now explore batch size
        'use_smote': [True, False]                             # Now explore SMOTE usage
    }
    
    best_params_final, _, best_metrics = tune_hyperparameters(
        X, y, phase3_param_grid, n_splits=n_splits, n_jobs=n_jobs
    )
    
    print("\nFinal best parameters:")
    for k, v in best_params_final.items():
        print(f"{k}: {v}")
    
    print("\nFinal best metrics:")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")
        
    return best_params_final

def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Train the final model using the best parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        best_params: Best parameters from tuning
        
    Returns:
        Trained model and test metrics
    """
    print("\nTraining final model with best parameters...")
    
    # Create model with best parameters
    final_model = TextNumericClassifier(
        max_words=best_params.get('max_words', 10000),
        max_sequence_length=100,  # Keep this fixed
        embedding_dim=best_params.get('embedding_dim', 100),
        l2_reg=best_params.get('l2_reg', 0.01)
    )
    
    # Train on full training set
    final_model.fit(
        X_train,
        y_train,
        use_smote=best_params.get('use_smote', True),
        pos_weight=best_params.get('pos_weight', 15.0),
        batch_size=best_params.get('batch_size', 32),
        epochs=30,  # Keep epochs fixed for final model
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    test_metrics = {
        'recall': test_recall,
        'precision': test_precision,
        'f1': test_f1,
        'threshold': final_model.threshold
    }
    
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_model.save_model(f'final_model_{timestamp}')
    
    return final_model, test_metrics

# Example usage
if __name__ == "__main__":
    # Load your data
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    X = train_df[['text', 'numeric_value']].values  # Assuming text and numeric columns
    y = train_df['label'].values
    X_test = test_df[['text', 'numeric_value']].values
    y_test = test_df['label'].values
    
    # Run phased hyperparameter tuning
    best_params = phased_tuning(X, y, n_splits=5, n_jobs=-1)
    
    # Train final model with best parameters
    final_model, test_metrics = train_final_model(X, y, X_test, y_test, best_params)
