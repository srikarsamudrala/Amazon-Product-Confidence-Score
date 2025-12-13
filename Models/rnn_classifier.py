"""
RNN Classifier for Amazon Reviews
==================================
Trains LSTM/GRU RNN models for spam detection and rating prediction.
Handles large datasets with chunked loading and efficient preprocessing.
"""

# Windows DLL fix for TensorFlow
import sys
if sys.platform == 'win32':
    import os
    # Add System32 to DLL search path (fixes TensorFlow DLL loading issues on Windows)
    if os.path.exists('C:/Windows/System32'):
        os.add_dll_directory('C:/Windows/System32')

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ReviewRNNClassifier:
    """
    RNN-based classifier for Amazon reviews.
    Supports spam detection (binary) and rating prediction (multi-class).
    """
    
    def __init__(
        self,
        input_file: str = "cleaned_dataset.csv",
        output_dir: str = "output/rnn_models",
        task: str = "spam",  # "spam" or "rating"
        max_words: int = 10000,
        max_length: int = 200,
        embedding_dim: int = 128,
        lstm_units: int = 64,
        batch_size: int = 64,
        epochs: int = 10,
        validation_split: float = 0.2,
        sample_size: Optional[int] = None  # None = use all data
    ):
        """
        Initialize RNN classifier.
        
        Args:
            input_file: Path to CSV file with reviews
            output_dir: Directory for saving models and results
            task: "spam" (binary) or "rating" (multi-class 1-5)
            max_words: Maximum vocabulary size
            max_length: Maximum sequence length (words per review)
            embedding_dim: Word embedding dimension
            lstm_units: Number of LSTM units
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            sample_size: Limit dataset size (None = use all)
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.sample_size = sample_size
        
        # Will be set during preprocessing
        self.tokenizer = None
        self.model = None
        self.history = None
        self.column_mapping = {}
        
        print(f"Initialized RNN Classifier")
        print(f"  Task: {task}")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_dir}")
    
    def _detect_columns(self, sample_df: pd.DataFrame) -> None:
        """Auto-detect column names."""
        print(f"Available columns: {list(sample_df.columns)}")
        for col in sample_df.columns:
            col_lower = col.lower()
            # Review text column - prioritize 'reviewtext', then 'review'
            if 'reviewtext' in col_lower:
                self.column_mapping['reviewText'] = col
            elif col_lower == 'review' and 'reviewText' not in self.column_mapping:
                self.column_mapping['reviewText'] = col
            # Rating column - prioritize 'rating', then 'overall'
            if 'rating' in col_lower and 'rating' not in self.column_mapping:
                self.column_mapping['rating'] = col
            elif col_lower == 'overall' and 'rating' not in self.column_mapping:
                self.column_mapping['rating'] = col
            # Spam column
            if 'spam' in col_lower or col_lower in ['class', 'is_spam']:
                self.column_mapping['is_spam'] = col
        print(f"Column mapping: {self.column_mapping}")
        
        # Verify we found review text
        if 'reviewText' not in self.column_mapping:
            raise ValueError(f"Review text column not found! Available columns: {list(sample_df.columns)}")
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)
        return text
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from CSV.
        
        Returns:
            Tuple of (X_texts, y_labels)
        """
        print("\nLoading data...")
        
        # Detect columns
        sample_df = pd.read_csv(self.input_file, nrows=100)
        self._detect_columns(sample_df)
        
        required_cols = ['reviewText', 'rating']
        if self.task == "spam":
            required_cols.append('is_spam')
        
        missing = [c for c in required_cols if c not in self.column_mapping]
        if missing:
            raise ValueError(f"Required columns not found: {missing}")
        
        # Load data in chunks
        texts = []
        labels = []
        chunk_size = 50000
        total_loaded = 0
        
        for chunk in pd.read_csv(
            self.input_file,
            chunksize=chunk_size,
            low_memory=False
        ):
            # Rename columns to standardized names
            chunk = chunk.rename(columns={v: k for k, v in self.column_mapping.items()})
            
            # Filter valid rows (now using standardized column name 'reviewText')
            chunk = chunk.dropna(subset=['reviewText'])
            chunk = chunk[chunk['reviewText'].astype(str).str.strip() != ""]
            
            # Clean text
            chunk['clean_text'] = chunk['reviewText'].apply(self._clean_text)
            chunk = chunk[chunk['clean_text'].str.len() > 10]  # Min 10 chars
            
            # Debug: show sample text
            if len(texts) == 0 and len(chunk) > 0:
                print(f"\nSample review text (first 100 chars): {chunk['clean_text'].iloc[0][:100]}...")
            
            # Extract labels based on task (using standardized column names after rename)
            if self.task == "spam":
                if 'is_spam' not in chunk.columns:
                    print("Warning: is_spam column not found, skipping chunk")
                    continue
                chunk['label'] = chunk['is_spam'].apply(
                    lambda x: 1 if pd.notna(x) and (int(x) == 1 or str(x).lower() in ['1', 'true', 'spam', 'yes']) else 0
                )
            else:  # rating prediction
                chunk['label'] = pd.to_numeric(chunk['rating'], errors='coerce')
                chunk = chunk[(chunk['label'] >= 1) & (chunk['label'] <= 5)]
                chunk['label'] = chunk['label'] - 1  # Convert to 0-4 for classification
            
            chunk = chunk.dropna(subset=['label'])
            
            texts.extend(chunk['clean_text'].tolist())
            labels.extend(chunk['label'].tolist())
            
            total_loaded += len(chunk)
            print(f"  Loaded {total_loaded:,} reviews...", end='\r')
            
            # Limit sample size if specified
            if self.sample_size and total_loaded >= self.sample_size:
                texts = texts[:self.sample_size]
                labels = labels[:self.sample_size]
                break
        
        print(f"\n✓ Loaded {len(texts):,} reviews")
        
        if not texts:
            raise ValueError("No valid reviews found in dataset")
        
        # Keep texts as list (don't convert to numpy array - saves memory)
        # Only convert labels to array
        y_labels = np.array(labels, dtype=int)
        
        # Print label distribution
        unique, counts = np.unique(y_labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
        
        # Return texts as list to avoid memory issues with large string arrays
        return texts, y_labels
    
    def preprocess_texts(self, texts) -> np.ndarray:
        """
        Tokenize and pad sequences.
        Processes in batches to avoid memory issues.
        
        Args:
            texts: List of text strings
            
        Returns:
            Padded sequences array
        """
        print("\nPreprocessing texts...")
        
        # Fit tokenizer on all texts (this is memory efficient)
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences in batches to save memory
        print("  Converting texts to sequences...")
        batch_size = 50000
        all_sequences = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            sequences = self.tokenizer.texts_to_sequences(batch)
            all_sequences.extend(sequences)
            if (i + batch_size) % 100000 == 0 or i + batch_size >= len(texts):
                print(f"    Processed {min(i + batch_size, len(texts)):,} / {len(texts):,} texts...", end='\r')
        
        print(f"\n  Padding sequences...")
        # Pad sequences
        X_padded = pad_sequences(all_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        print(f"✓ Preprocessed {len(X_padded):,} sequences")
        print(f"  Vocabulary size: {len(self.tokenizer.word_index):,}")
        print(f"  Sequence shape: {X_padded.shape}")
        
        # Save tokenizer
        tokenizer_path = self.output_dir / 'tokenizer.pkl'
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"  Saved tokenizer: {tokenizer_path}")
        
        return X_padded
    
    def build_model(self, num_classes: int) -> keras.Model:
        """
        Build LSTM-based RNN model.
        
        Args:
            num_classes: Number of output classes (2 for spam, 5 for rating)
            
        Returns:
            Compiled Keras model
        """
        print(f"\nBuilding RNN model ({num_classes} classes)...")
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding'
            ),
            
            # LSTM layer with dropout
            layers.LSTM(
                units=self.lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False,
                name='lstm'
            ),
            
            # Dense layers
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(0.3, name='dropout_1'),
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Output layer
            layers.Dense(
                num_classes,
                activation='softmax' if num_classes > 2 else 'sigmoid',
                name='output'
            )
        ])
        
        # Compile model
        if num_classes == 1:  # Binary classification
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:  # Multi-class classification
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        # Build model with sample input shape to enable count_params
        model.build(input_shape=(None, self.max_length))
        
        print(f"✓ Model built")
        print(f"  Total parameters: {model.count_params():,}")
        model.summary()
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> keras.callbacks.History:
        """
        Train the RNN model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        print("\nTraining model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.keras'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        if X_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training completed")
        
        # Save final model
        final_model_path = self.output_dir / 'final_model.keras'
        self.model.save(final_model_path)
        print(f"  Saved model: {final_model_path}")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating model...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, batch_size=self.batch_size, verbose=0)
        
        if self.task == "spam":
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        if self.task == "spam":
            try:
                # For binary classification, y_pred_proba might be 2D, flatten it
                y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
                auc = roc_auc_score(y_test, y_pred_proba_flat)
                metrics['roc_auc'] = auc
            except:
                pass
        
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        print(f"✓ Test F1 Score: {f1:.4f}")
        if 'roc_auc' in metrics:
            print(f"✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save metrics
        metrics_path = self.output_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 40 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"  Saved metrics: {metrics_path}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            return
        
        print("\nGenerating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_path = self.output_dir / 'training_history.png'
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {history_path}")
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title(f'Confusion Matrix - {self.task.title()} Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {cm_path}")
    
    def plot_roc_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray):
        """Plot ROC curve (for binary classification only)."""
        if self.task != "spam":
            return
        
        # Flatten predictions for binary classification
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_flat)
        auc = roc_auc_score(y_test, y_pred_proba_flat)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Spam Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = self.output_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {roc_path}")
    
    def run_full_pipeline(self):
        """Run complete pipeline: load, preprocess, train, evaluate."""
        print("=" * 60)
        print("RNN Classification Pipeline")
        print("=" * 60)
        
        # Load data
        X_texts, y_labels = self.load_data()
        
        # Preprocess
        X_sequences = self.preprocess_texts(X_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_labels,
            test_size=0.2,
            random_state=42,
            stratify=y_labels if len(np.unique(y_labels)) > 1 else None
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.validation_split,
            random_state=42,
            stratify=y_train if len(np.unique(y_train)) > 1 else None
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train):,}")
        print(f"  Validation: {len(X_val):,}")
        print(f"  Test: {len(X_test):,}")
        
        # Build model
        # For binary classification (spam), use 1 output unit, not 2
        if self.task == "spam":
            num_classes = 1  # Binary classification: single output with sigmoid
        else:
            num_classes = len(np.unique(y_labels))  # Multi-class: one output per class
        self.model = self.build_model(num_classes)
        
        # Train
        self.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        # Predictions for plots
        y_pred_proba = self.model.predict(X_test, batch_size=self.batch_size, verbose=0)
        if self.task == "spam":
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Generate plots
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred)
        if self.task == "spam":
            self.plot_roc_curve(y_test, y_pred_proba)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}/")


def main():
    """
    Main execution function.
    
    To run independently:
    1. Ensure cleaned_dataset.csv is in the same directory
    2. Install dependencies: pip install -r requirements.txt
    3. Run: python rnn_classifier.py
    
    For faster testing, set SAMPLE_SIZE to a smaller number (e.g., 50000)
    For full dataset, set SAMPLE_SIZE = None
    """
    # ==================== CONFIGURATION ====================
    INPUT_FILE = "cleaned_dataset.csv"  # Your CSV file with review text
    TASK = "spam"  # "spam" (binary) or "rating" (multi-class 1-5)
    SAMPLE_SIZE = None  # None = use ALL data, or set limit for faster testing (recommended: 50000 for testing)
    MAX_WORDS = 10000  # Vocabulary size
    MAX_LENGTH = 100  # Max words per review
    EMBEDDING_DIM = 128  # Word embedding dimension
    LSTM_UNITS = 32  # LSTM layer size
    BATCH_SIZE = 256  # Training batch size
    EPOCHS = 5  # Number of training epochs
    # =======================================================
    
    print("=" * 70)
    print("RNN Classifier for Amazon Reviews - Standalone Execution")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Task: {TASK}")
    print(f"  Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL DATA'}")
    print(f"  Max words: {MAX_WORDS}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("\nStarting pipeline...\n")
    
    try:
        classifier = ReviewRNNClassifier(
            input_file=INPUT_FILE,
            task=TASK,
            sample_size=SAMPLE_SIZE,
            max_words=MAX_WORDS,
            max_length=MAX_LENGTH,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )
        
        classifier.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("SUCCESS! Training completed.")
        print("=" * 70)
        print(f"\nOutput files saved to: output/rnn_models/")
        print("  - best_model.keras (best model during training)")
        print("  - final_model.keras (final trained model)")
        print("  - tokenizer.pkl (text tokenizer)")
        print("  - metrics.txt (evaluation metrics)")
        print("  - training_history.png (loss/accuracy plots)")
        print("  - confusion_matrix.png (confusion matrix)")
        if TASK == "spam":
            print("  - roc_curve.png (ROC curve)")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found: {e}")
        print("   Make sure 'cleaned_dataset.csv' is in the current directory.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

