import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import backend as K
from imblearn.over_sampling import SMOTE

class TextNumericClassifier:
    def __init__(self, max_words=10000, max_sequence_length=100, embedding_dim=100):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.scaler = None
        self.model = None
        self.threshold = 0.5
        
    def _extract_features(self, data_series):
        """Extract text and numeric features from mixed data"""
        texts = []
        numerics = []
        
        for item in data_series:
            # Extract numbers using regex
            numbers = re.findall(r'\d+\.?\d*', str(item))
            numeric_values = [float(num) for num in numbers] if numbers else [0.0]
            
            # Get text part by removing numbers
            text_part = re.sub(r'\d+\.?\d*', '', str(item)).strip()
            
            texts.append(text_part)
            numerics.append(np.mean(numeric_values))  # Using mean as a simple aggregation
        
        return texts, np.array(numerics).reshape(-1, 1)
    
    def preprocess_data(self, X_data):
        """Preprocess mixed text and numeric data"""
        texts, numerics = self._extract_features(X_data)
        
        # Process text data
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_words)
            self.tokenizer.fit_on_texts(texts)
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Process numeric data
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_numerics = self.scaler.fit_transform(numerics)
        else:
            scaled_numerics = self.scaler.transform(numerics)
            
        return padded_sequences, scaled_numerics
    
    def create_model(self):
        """Create a deep neural network for text and numeric data"""
        # Get actual vocabulary size
        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_words)
        
        # Text input branch
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        embedding = Embedding(vocab_size, self.embedding_dim)(text_input)
        lstm = LSTM(128, dropout=0.3, recurrent_dropout=0.3)(embedding)
        
        # Numeric input branch
        numeric_input = Input(shape=(1,), name='numeric_input')
        
        # Combine text and numeric features
        combined = Concatenate()([lstm, numeric_input])
        
        # Dense layers with regularization to prevent overfitting
        dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dropout1)
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

    def weighted_binary_crossentropy(self, pos_weight=10.0):
        """Create weighted binary crossentropy loss function for higher recall
        
        Args:
            pos_weight: Weight for positive class (higher values improve recall)
        """
        def loss(y_true, y_pred):
            # Clip prediction values to avoid log(0)
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            
            # Calculate loss with different weights for positive and negative classes
            pos_loss = -y_true * K.log(y_pred) * pos_weight
            neg_loss = -(1 - y_true) * K.log(1 - y_pred)
            
            return K.mean(pos_loss + neg_loss)
        
        return loss
    
    def _create_recall_focused_loss(self):
        """Custom loss function focused on maximizing recall"""
        def recall_focused_loss(y_true, y_pred):
            # Extract positive and negative examples
            pos = K.cast(y_true > 0, K.floatx())
            neg = 1 - pos
            
            # Calculate losses for positive and negative examples (heavily weight positives)
            pos_loss = -K.mean(pos * K.log(K.clip(y_pred, K.epsilon(), 1.0))) * 20.0  # Much higher weight
            neg_loss = -K.mean(neg * K.log(K.clip(1 - y_pred, K.epsilon(), 1.0)))
            
            return pos_loss + neg_loss
        
        return recall_focused_loss
    
    def fit(self, X_train, y_train, use_smote=True, pos_weight=15.0, epochs=30, batch_size=32, validation_split=0.2):
        """Train the model with high recall focus
        
        Args:
            X_train: Training features (mixed text and numeric)
            y_train: Training labels (binary)
            use_smote: Whether to use SMOTE oversampling (default: True)
            pos_weight: Weight for positive class in weighted cross-entropy (default: 15.0)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
        """
        # Preprocess data
        train_text_data, train_numerics_scaled = self.preprocess_data(X_train)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
            
        # Use weighted cross-entropy loss
        weighted_loss = self.weighted_binary_crossentropy(pos_weight=pos_weight)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_loss,
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Calculate class weights to emphasize positive class
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Make positive class even more important
        positive_class_idx = 1  # Assuming 1 is positive class
        class_weight_dict[positive_class_idx] *= 5  # Additional class weighting
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_recall', patience=5, mode='max', restore_best_weights=True),
            ModelCheckpoint('best_recall_model.h5', monitor='val_recall', mode='max', save_best_only=True)
        ]
        
        # Apply SMOTE if requested
        if use_smote:
            # First split validation data to avoid data leakage
            X_text_train, X_text_val, X_num_train, X_num_val, y_tr, y_val = train_test_split(
                train_text_data, train_numerics_scaled, y_train, test_size=validation_split
            )
            
            # Apply SMOTE to training portion only
            smote = SMOTE(sampling_strategy=1.0, random_state=42)
            
            # We need to keep track of indices for text data
            indices = np.arange(len(y_tr))
            # Reshape for SMOTE
            indices_2d = indices.reshape(-1, 1)
            
            # Combine numeric features with indices for SMOTE
            combined_features = np.hstack([X_num_train, indices_2d])
            resampled_features, y_resampled = smote.fit_resample(combined_features, y_tr)
            
            # Extract resampled numeric features and indices
            X_num_resampled = resampled_features[:, :-1]
            indices_resampled = resampled_features[:, -1].astype(int)
            
            # Handle text data for original samples
            X_text_resampled = []
            for idx in indices_resampled:
                if idx < len(X_text_train):
                    X_text_resampled.append(X_text_train[idx])
                else:
                    # For synthetic samples, use the text from a random sample of same class
                    # Find a random sample from the same class to duplicate for text
                    class_indices = [i for i, y in enumerate(y_tr) if y == y_resampled[idx]]
                    if class_indices:
                        random_idx = np.random.choice(class_indices)
                        X_text_resampled.append(X_text_train[random_idx])
            
            X_text_resampled = np.array(X_text_resampled)
            
            # Train with resampled data
            history = self.model.fit(
                [X_text_resampled, X_num_resampled], 
                y_resampled,
                validation_data=([X_text_val, X_num_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
        else:
            # Train without SMOTE
            history = self.model.fit(
                [train_text_data, train_numerics_scaled], 
                y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
        
        # Find optimal threshold for 100% recall
        self._find_optimal_threshold(train_text_data, train_numerics_scaled, y_train)
        
        return history
    
    def _find_optimal_threshold(self, X_text, X_numeric, y_true):
        """Find threshold that gives 100% recall on training data"""
        y_pred_prob = self.model.predict([X_text, X_numeric])
        
        best_threshold = 0.01  # Start with very low threshold
        best_recall = 0
        
        # Try progressively lower thresholds until we hit perfect recall
        for threshold in np.arange(0.5, 0.01, -0.01):
            y_pred = (y_pred_prob >= threshold).astype(int)
            recall = recall_score(y_true, y_pred)
            
            if recall >= 0.999:  # Account for numerical precision
                best_threshold = threshold
                best_recall = recall
                break
                
        # If we haven't hit perfect recall, use the minimum threshold
        if best_recall < 0.999:
            best_threshold = 0.01
            y_pred = (y_pred_prob >= best_threshold).astype(int)
            best_recall = recall_score(y_true, y_pred)
            
        print(f"Optimal threshold: {best_threshold}, Expected recall: {best_recall}")
        self.threshold = best_threshold
        return best_threshold
    
    def predict(self, X_data):
        """Make predictions with optimal threshold for high recall"""
        text_data, numeric_data = self.preprocess_data(X_data)
        predictions_prob = self.model.predict([text_data, numeric_data])
        predictions = (predictions_prob >= self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, X_data):
        """Return probability predictions"""
        text_data, numeric_data = self.preprocess_data(X_data)
        return self.model.predict([text_data, numeric_data])
    
    def evaluate(self, X_data, y_true):
        """Evaluate model with optimal threshold"""
        text_data, numeric_data = self.preprocess_data(X_data)
        y_pred_prob = self.model.predict([text_data, numeric_data])
        y_pred = (y_pred_prob >= self.threshold).astype(int)
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # Calculate and print recall specifically
        recall = recall_score(y_true, y_pred)
        print(f"Recall: {recall:.4f}")
        
        return y_pred
    
    def save_model(self, filepath):
        """Save the model and related components"""
        self.model.save(filepath)
        
        # Save tokenizer and scaler with joblib
        import joblib
        joblib.dump(self.tokenizer, filepath + '_tokenizer.pkl')
        joblib.dump(self.scaler, filepath + '_scaler.pkl')
        
        # Save threshold
        with open(filepath + '_threshold.txt', 'w') as f:
            f.write(str(self.threshold))
    
    @classmethod
    def load_model(cls, filepath):
        """Load saved model and components"""
        import joblib
        
        instance = cls()
        instance.model = load_model(filepath, custom_objects={
            'loss': instance.weighted_binary_crossentropy()
        })
        instance.tokenizer = joblib.load(filepath + '_tokenizer.pkl')
        instance.scaler = joblib.load(filepath + '_scaler.pkl')
        
        with open(filepath + '_threshold.txt', 'r') as f:
            instance.threshold = float(f.read().strip())
        
        return instance

# Example usage
if __name__ == "__main__":
    # Load your data
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    # Assuming columns are named 'text_numeric' and 'label'
    X_train = train_df['text_numeric'].values
    y_train = train_df['label'].values
    X_test = test_df['text_numeric'].values
    y_test = test_df['label'].values
    
    # Create and train classifier with weighted cross-entropy
    classifier = TextNumericClassifier(max_words=10000, max_sequence_length=100)
    classifier.fit(
        X_train, 
        y_train, 
        use_smote=True, 
        pos_weight=15.0,  # Weight for positive class in loss function
        epochs=30, 
        batch_size=32
    )
    
    # Evaluate on test data
    y_pred = classifier.evaluate(X_test, y_test)
    
    # Save the model
    classifier.save_model('high_recall_classifier')
