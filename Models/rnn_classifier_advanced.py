"""
Advanced RNN Spam Classifier
=============================
Upgraded RNN classifier with Bidirectional LSTM, Attention, GloVe embeddings,
metadata features, and advanced training features.

Target: 92-95% accuracy, spam F1>0.90, non-spam F1>0.93
"""

# Windows DLL fix for TensorFlow
import sys
if sys.platform == 'win32':
    import os
    if os.path.exists('C:/Windows/System32'):
        os.add_dll_directory('C:/Windows/System32')

import pandas as pd
import numpy as np
import os
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
from datetime import datetime, timedelta
import time
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== HARDWARE OPTIMIZATION ====================
# Configure TensorFlow for maximum performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable Intel optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging

# GPU Configuration (Intel Arc)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU acceleration enabled: {len(physical_devices)} GPU(s)")
        # Enable mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[OK] Mixed precision training enabled (float16)")
    except RuntimeError as e:
        print(f"[WARN] GPU configuration error: {e}")
else:
    print("[INFO] No GPU detected, using CPU")

# CPU Threading - Use all 22 logical cores
tf.config.threading.set_intra_op_parallelism_threads(22)
tf.config.threading.set_inter_op_parallelism_threads(22)
print(f"[OK] CPU threading: 22 threads (all logical cores)")

# Enable optimizations
try:
    tf.config.optimizer.set_jit(True)  # XLA JIT compilation
    print("[OK] XLA JIT compilation enabled")
except:
    print("[INFO] XLA JIT not available, continuing without it")
# ================================================================


class LiveProgressCallback(Callback):
    """Custom callback for live training progress updates."""
    def __init__(self, total_epochs, start_time=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = start_time or time.time()
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{self.total_epochs} - Starting...")
        print(f"{'='*70}")
        sys.stdout.flush()
        
    def on_batch_end(self, batch, logs=None):
        # Update every 10 batches to avoid spam
        if batch % 10 == 0:
            elapsed = time.time() - self.epoch_start
            loss = logs.get('loss', 0)
            acc = logs.get('accuracy', 0)
            print(f"  Batch {batch}: Loss={loss:.4f}, Acc={acc:.4f}, Time={elapsed:.1f}s", end='\r')
            sys.stdout.flush()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        total_elapsed = time.time() - self.start_time
        
        # Calculate ETA
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = timedelta(seconds=int(eta_seconds))
        
        print(f"\n  Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f"  Accuracy: {logs.get('accuracy', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        if 'val_f1' in logs:
            print(f"  Val F1: {logs['val_f1']:.4f}")
        print(f"  Total time: {timedelta(seconds=int(total_elapsed))} | ETA: {eta}")
        print(f"{'='*70}\n")
        sys.stdout.flush()


class F1ScoreCallback(Callback):
    """Custom callback to compute F1 score during training."""
    def __init__(self, validation_data, task='spam', use_metadata=False):
        super().__init__()
        self.validation_data = validation_data
        self.task = task
        self.use_metadata = use_metadata
        self.best_f1 = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        print("  Computing F1 score on validation set...", end='\r')
        sys.stdout.flush()
        
        if self.use_metadata:
            X_val, metadata_val, y_val = self.validation_data
            val_data = [X_val, metadata_val]
        else:
            X_val, y_val = self.validation_data
            val_data = X_val
        
        y_pred_proba = self.model.predict(val_data, verbose=0, batch_size=256)
        
        if self.task == "spam":
            y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
            y_pred = (y_pred_proba_flat > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        f1 = f1_score(y_val, y_pred, average='weighted')
        logs['val_f1'] = f1
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            print(f"  ✓ Epoch {epoch+1}: New best F1 = {f1:.4f} (improved!)")
        else:
            print(f"  Epoch {epoch+1}: F1 = {f1:.4f} (best: {self.best_f1:.4f})")
        sys.stdout.flush()


class PlotCallback(Callback):
    """Callback to save training history plots after each epoch."""
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history_dict = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        """Generate and save plot after each epoch."""
        if logs is None:
            logs = {}
            
        # Store metrics
        for key in self.history_dict.keys():
            if key in logs:
                self.history_dict[key].append(logs[key])
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss
            if self.history_dict['loss']:
                axes[0, 0].plot(self.history_dict['loss'], label='Train Loss', marker='o')
            if self.history_dict['val_loss']:
                axes[0, 0].plot(self.history_dict['val_loss'], label='Val Loss', marker='s')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy
            if self.history_dict['accuracy']:
                axes[0, 1].plot(self.history_dict['accuracy'], label='Train Accuracy', marker='o')
            if self.history_dict['val_accuracy']:
                axes[0, 1].plot(self.history_dict['val_accuracy'], label='Val Accuracy', marker='s')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision
            if self.history_dict['precision']:
                axes[1, 0].plot(self.history_dict['precision'], label='Train Precision', marker='o')
            if self.history_dict['val_precision']:
                axes[1, 0].plot(self.history_dict['val_precision'], label='Val Precision', marker='s')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Recall
            if self.history_dict['recall']:
                axes[1, 1].plot(self.history_dict['recall'], label='Train Recall', marker='o')
            if self.history_dict['val_recall']:
                axes[1, 1].plot(self.history_dict['val_recall'], label='Val Recall', marker='s')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            history_path = self.output_dir / 'training_history.png'
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n[OK] Saved training history plot: {history_path}")
            print(f"  Epoch {epoch+1} metrics: loss={logs.get('loss', 'N/A'):.4f}, "
                  f"accuracy={logs.get('accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Could not save plot at epoch {epoch+1}: {e}")
            print(traceback.format_exc())


class AttentionLayer(layers.Layer):
    """Attention mechanism for LSTM outputs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_dim)
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1))  # (batch_size, seq_len, 1)
        a = tf.nn.softmax(e, axis=1)  # (batch_size, seq_len, 1)
        output = tf.reduce_sum(a * inputs, axis=1)  # (batch_size, hidden_dim)
        return output
    
    def get_config(self):
        return super().get_config()


class AdvancedSpamClassifier:
    """
    Advanced RNN-based spam classifier with:
    - Bidirectional LSTM (2 layers: 128→64 units)
    - Attention mechanism
    - Pre-trained GloVe embeddings
    - Metadata features
    - Class weights and threshold optimization
    """
    
    def __init__(
        self,
        input_file: str = "cleaned_dataset.csv",
        output_dir: str = "output/rnn_models_advanced",
        max_words: int = 20000,
        max_length: int = 150,
        embedding_dim: int = 200,  # GloVe 200d
        batch_size: int = 128,
        epochs: int = 20,
        validation_split: float = 0.2,
        sample_size: Optional[int] = None,
        use_pretrained_embeddings: bool = True,
        use_attention: bool = True,
        use_metadata_features: bool = True,
        optimize_threshold: bool = True
    ):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.sample_size = sample_size
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_attention = use_attention
        self.use_metadata_features = use_metadata_features
        self.optimize_threshold = optimize_threshold
        
        # Will be set during processing
        self.tokenizer = None
        self.embedding_matrix = None
        self.model = None
        self.history = None
        self.column_mapping = {}
        self.optimal_threshold = 0.5
        
        print("=" * 70)
        print("Advanced RNN Spam Classifier")
        print("=" * 70)
        print(f"  Input file: {input_file}")
        print(f"  Output directory: {output_dir}")
        print(f"  Max words: {max_words}")
        print(f"  Max length: {max_length}")
        print(f"  Use GloVe embeddings: {use_pretrained_embeddings}")
        print(f"  Use attention: {use_attention}")
        print(f"  Use metadata features: {use_metadata_features}")
        print(f"  Optimize threshold: {optimize_threshold}")
    
    def download_glove_embeddings(self) -> str:
        """Download GloVe 200d embeddings if not present."""
        glove_dir = Path("glove_embeddings")
        glove_dir.mkdir(exist_ok=True)
        glove_file = glove_dir / "glove.6B.200d.txt"
        
        if glove_file.exists():
            print(f"[OK] GloVe embeddings found: {glove_file}")
            return str(glove_file)
        
        print("Downloading GloVe 200d embeddings (822MB)...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = glove_dir / "glove.6B.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=self._download_progress)
            print(f"\n[OK] Downloaded: {zip_path}")
            
            print("Extracting GloVe embeddings...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("glove.6B.200d.txt", glove_dir)
            
            # Clean up zip file
            zip_path.unlink()
            print(f"[OK] Extracted: {glove_file}")
            return str(glove_file)
        except Exception as e:
            print(f"[WARNING] Could not download GloVe embeddings: {e}")
            print("  Will use random embeddings instead.")
            return None
    
    def _download_progress(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r  Downloading: {percent:.1f}%", end='', flush=True)
    
    def load_glove_embeddings(self, glove_file: str) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings into dictionary."""
        print("\nLoading GloVe embeddings...")
        embeddings_index = {}
        
        try:
            with open(glove_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            
            print(f"[OK] Loaded {len(embeddings_index):,} word vectors")
            return embeddings_index
        except Exception as e:
            print(f"[ERROR] Error loading GloVe: {e}")
            return {}
    
    def create_embedding_matrix(self, word_index: Dict[str, int], embeddings_index: Dict[str, np.ndarray]) -> np.ndarray:
        """Create embedding matrix from GloVe vectors."""
        print("\nCreating embedding matrix...")
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        hits = 0
        misses = 0
        
        for word, i in word_index.items():
            if i >= self.max_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        
        print(f"  Found embeddings for {hits:,} words ({hits/(hits+misses)*100:.1f}%)")
        print(f"  Missing embeddings for {misses:,} words")
        
        return embedding_matrix
    
    def _detect_columns(self, sample_df: pd.DataFrame) -> None:
        """Auto-detect column names."""
        print(f"Available columns: {list(sample_df.columns)}")
        for col in sample_df.columns:
            col_lower = col.lower()
            if 'reviewtext' in col_lower:
                self.column_mapping['reviewText'] = col
            elif col_lower == 'review' and 'reviewText' not in self.column_mapping:
                self.column_mapping['reviewText'] = col
            if ('rating' in col_lower or col_lower == 'overall') and 'rating' not in self.column_mapping:
                self.column_mapping['rating'] = col
            if 'asin' in col_lower and 'asin' not in self.column_mapping:
                self.column_mapping['asin'] = col
            if 'spam' in col_lower or col_lower in ['class', 'is_spam']:
                self.column_mapping['is_spam'] = col
        print(f"Column mapping: {self.column_mapping}")
        
        if 'reviewText' not in self.column_mapping:
            raise ValueError(f"Review text column not found! Available: {list(sample_df.columns)}")
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'http\S+', '', text)
        return text
    
    def _extract_metadata_features(self, texts: list, ratings: Optional[list] = None) -> np.ndarray:
        """Extract metadata features: length, capital ratio, punctuation density, rating."""
        print("\nExtracting metadata features...")
        sys.stdout.flush()
        features = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            text_str = str(text)
            # Review length (normalized)
            length = len(text_str.split())
            length_norm = min(length / 500.0, 1.0)  # Normalize to [0, 1], cap at 500 words
            
            # Capital ratio
            capitals = sum(1 for c in text_str if c.isupper())
            capital_ratio = capitals / max(len(text_str), 1)
            
            # Punctuation density
            punct = sum(1 for c in text_str if c in '!?.')
            punct_density = punct / max(len(text_str.split()), 1)
            
            # Rating (if available)
            rating = ratings[i] if ratings and i < len(ratings) else 3.0
            rating_norm = (rating - 1.0) / 4.0  # Normalize 1-5 to 0-1
            
            features.append([length_norm, capital_ratio, punct_density, rating_norm])
            
            # Progress update every 10k texts
            if (i + 1) % 10000 == 0 or (i + 1) == len(texts):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Processed {i+1:,} / {len(texts):,} texts ({((i+1)/len(texts)*100):.1f}%) | "
                      f"Speed: {rate:.0f} texts/sec", end='\r')
                sys.stdout.flush()
        
        total_time = time.time() - start_time
        print(f"\n  ✓ Extracted metadata for {len(features):,} texts in {total_time:.1f}s")
        sys.stdout.flush()
        
        return np.array(features, dtype=np.float32)
    
    def load_data(self) -> Tuple[list, np.ndarray, Optional[np.ndarray]]:
        """Load and preprocess data from CSV."""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        start_time = time.time()
        
        sample_df = pd.read_csv(self.input_file, nrows=100)
        self._detect_columns(sample_df)
        
        texts = []
        labels = []
        ratings = []
        chunk_size = 50000
        total_loaded = 0
        chunk_num = 0
        
        print("  Reading CSV file in chunks...")
        sys.stdout.flush()
        
        for chunk in pd.read_csv(
            self.input_file,
            chunksize=chunk_size,
            low_memory=False,
            engine='c'  # Use C engine for faster parsing
        ):
            chunk_num += 1
            chunk_start = time.time()
            
            chunk = chunk.rename(columns={v: k for k, v in self.column_mapping.items()})
            
            chunk = chunk.dropna(subset=['reviewText'])
            chunk = chunk[chunk['reviewText'].astype(str).str.strip() != ""]
            
            chunk['clean_text'] = chunk['reviewText'].apply(self._clean_text)
            chunk = chunk[chunk['clean_text'].str.len() > 10]
            
            # Extract labels
            if 'is_spam' not in chunk.columns:
                print(f"  Warning: is_spam column not found, skipping chunk {chunk_num}")
                continue
            
            chunk['label'] = chunk['is_spam'].apply(
                lambda x: 1 if pd.notna(x) and (int(x) == 1 or str(x).lower() in ['1', 'true', 'spam', 'yes']) else 0
            )
            
            # Extract ratings if available
            if 'rating' in chunk.columns:
                chunk['rating_val'] = pd.to_numeric(chunk['rating'], errors='coerce')
                chunk['rating_val'] = chunk['rating_val'].fillna(3.0)
            else:
                chunk['rating_val'] = 3.0
            
            chunk = chunk.dropna(subset=['label'])
            
            texts.extend(chunk['clean_text'].tolist())
            labels.extend(chunk['label'].tolist())
            ratings.extend(chunk['rating_val'].tolist())
            
            total_loaded += len(chunk)
            chunk_time = time.time() - chunk_start
            elapsed = time.time() - start_time
            rate = total_loaded / elapsed if elapsed > 0 else 0
            
            print(f"  Chunk {chunk_num}: Loaded {total_loaded:,} reviews | "
                  f"Speed: {rate:.0f} reviews/sec | "
                  f"Time: {elapsed:.1f}s", end='\r')
            sys.stdout.flush()
            
            if self.sample_size and total_loaded >= self.sample_size:
                texts = texts[:self.sample_size]
                labels = labels[:self.sample_size]
                ratings = ratings[:self.sample_size]
                break
        
        total_time = time.time() - start_time
        print(f"\n  ✓ Loaded {len(texts):,} reviews in {total_time:.1f}s ({len(texts)/total_time:.0f} reviews/sec)")
        
        if not texts:
            raise ValueError("No valid reviews found in dataset")
        
        y_labels = np.array(labels, dtype=int)
        y_ratings = np.array(ratings, dtype=float) if self.use_metadata_features else None
        
        unique, counts = np.unique(y_labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique, counts))}")
        print("="*70)
        sys.stdout.flush()
        
        return texts, y_labels, y_ratings
    
    def preprocess_texts(self, texts: list) -> np.ndarray:
        """Tokenize and pad sequences."""
        print("\n" + "="*70)
        print("PREPROCESSING TEXTS")
        print("="*70)
        start_time = time.time()
        
        print("  Fitting tokenizer on texts...")
        sys.stdout.flush()
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        print(f"  ✓ Tokenizer fitted: {len(self.tokenizer.word_index):,} unique words")
        
        print("  Converting texts to sequences...")
        sys.stdout.flush()
        batch_size = 50000
        all_sequences = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_start = time.time()
            sequences = self.tokenizer.texts_to_sequences(batch)
            all_sequences.extend(sequences)
            processed = min(i + batch_size, len(texts))
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"    Processed {processed:,} / {len(texts):,} texts ({processed/len(texts)*100:.1f}%) | "
                  f"Speed: {rate:.0f} texts/sec", end='\r')
            sys.stdout.flush()
        
        print(f"\n  Padding sequences to max_length={self.max_length}...")
        sys.stdout.flush()
        X_padded = pad_sequences(all_sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        total_time = time.time() - start_time
        print(f"  ✓ Preprocessed {len(X_padded):,} sequences in {total_time:.1f}s")
        print(f"  Vocabulary size: {len(self.tokenizer.word_index):,}")
        print(f"  Sequence shape: {X_padded.shape}")
        
        # Save tokenizer
        tokenizer_path = self.output_dir / 'tokenizer.pkl'
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"  ✓ Saved tokenizer: {tokenizer_path}")
        print("="*70)
        sys.stdout.flush()
        
        return X_padded
    
    def build_model(self, num_classes: int = 1, metadata_dim: int = 4) -> Model:
        """Build advanced model with Bidirectional LSTM, Attention, and Metadata."""
        print(f"\nBuilding advanced RNN model...")
        
        # Text input
        text_input = Input(shape=(self.max_length,), name='text_input')
        
        # Embedding layer
        if self.use_pretrained_embeddings and self.embedding_matrix is not None:
            embedding_layer = layers.Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                weights=[self.embedding_matrix],
                trainable=False,  # Freeze GloVe embeddings
                name='embedding'
            )
            print("  Using pre-trained GloVe embeddings (frozen)")
        else:
            embedding_layer = layers.Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name='embedding'
            )
            print("  Using trainable embeddings")
        
        embedded = embedding_layer(text_input)
        
        # Bidirectional LSTM layers
        lstm1 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm_1'
        )(embedded)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=self.use_attention, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm_2'
        )(lstm1)
        
        # Attention mechanism
        if self.use_attention:
            attention_output = AttentionLayer(name='attention')(lstm2)
            text_features = attention_output
            print("  Using attention mechanism")
        else:
            text_features = lstm2
            print("  No attention mechanism")
        
        # Metadata features (if enabled)
        if self.use_metadata_features:
            metadata_input = Input(shape=(metadata_dim,), name='metadata_input')
            metadata_dense = layers.Dense(32, activation='relu', name='metadata_dense')(metadata_input)
            metadata_dropout = layers.Dropout(0.3, name='metadata_dropout')(metadata_dense)
            
            # Concatenate text and metadata features
            combined = layers.Concatenate(name='concat')([text_features, metadata_dropout])
            print("  Using metadata features")
        else:
            combined = text_features
            metadata_input = None
        
        # Dense layers
        dense1 = layers.Dense(128, activation='relu', name='dense_1')(combined)
        dropout1 = layers.Dropout(0.4, name='dropout_1')(dense1)
        dense2 = layers.Dense(64, activation='relu', name='dense_2')(dropout1)
        dropout2 = layers.Dropout(0.3, name='dropout_2')(dense2)
        
        # Output layer (use float32 for final layer in mixed precision)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            output = layers.Dense(num_classes, activation='sigmoid', 
                                 dtype='float32', name='output')(dropout2)
        else:
            output = layers.Dense(num_classes, activation='sigmoid', name='output')(dropout2)
        
        # Create model
        if self.use_metadata_features:
            self.model = Model(inputs=[text_input, metadata_input], outputs=output, name='advanced_spam_classifier')
        else:
            self.model = Model(inputs=text_input, outputs=output, name='advanced_spam_classifier')
        
        # Compile with mixed precision support
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # Wrap optimizer for mixed precision if needed
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Build model
        if self.use_metadata_features:
            self.model.build([(None, self.max_length), (None, metadata_dim)])
        else:
            self.model.build(input_shape=(None, self.max_length))
        
        print(f"[OK] Model built")
        print(f"  Total parameters: {self.model.count_params():,}")
        self.model.summary()
        
        return self.model
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"\nClass weights: {class_weights}")
        return class_weights
    
    def optimize_threshold(self, X_val, y_val, metadata_val=None) -> float:
        """Find optimal decision threshold using F1 score."""
        print("\nOptimizing decision threshold...")
        sys.stdout.flush()
        
        if self.use_metadata_features and metadata_val is not None:
            val_data = [X_val, metadata_val]
        else:
            val_data = X_val
        
        print("  Computing predictions on validation set...")
        sys.stdout.flush()
        y_pred_proba = self.model.predict(val_data, batch_size=self.batch_size, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        thresholds = np.arange(0.3, 0.8, 0.01)
        print(f"  Testing {len(thresholds)} threshold values...")
        sys.stdout.flush()
        for i, threshold in enumerate(thresholds):
            y_pred = (y_pred_proba_flat > threshold).astype(int)
            f1 = f1_score(y_val, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            if (i + 1) % 10 == 0:
                print(f"    Tested {i+1}/{len(thresholds)} thresholds | Best F1: {best_f1:.4f} @ {best_threshold:.3f}", end='\r')
                sys.stdout.flush()
        
        print(f"\n  ✓ Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        sys.stdout.flush()
        return best_threshold
    
    def train(
        self,
        X_train,
        y_train: np.ndarray,
        X_val=None,
        y_val: Optional[np.ndarray] = None,
        metadata_train=None,
        metadata_val=None
    ) -> keras.callbacks.History:
        """Train the model."""
        print("\nTraining model...")
        
        # Class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_f1' if y_val is not None else 'f1',
                patience=5,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.keras'),
                monitor='val_f1' if y_val is not None else 'f1',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if y_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add live progress callback
        live_callback = LiveProgressCallback(total_epochs=self.epochs, start_time=time.time())
        callbacks.append(live_callback)
        
        # Add F1 callback
        if y_val is not None:
            if self.use_metadata_features:
                f1_callback = F1ScoreCallback((X_val, metadata_val, y_val), task='spam', use_metadata=True)
            else:
                f1_callback = F1ScoreCallback((X_val, y_val), task='spam', use_metadata=False)
            callbacks.append(f1_callback)
        
        # Add plot callback to save plots after each epoch
        plot_callback = PlotCallback(self.output_dir)
        callbacks.append(plot_callback)
        
        # Add checkpoint callback that saves after every epoch (not just best)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(self.output_dir / 'checkpoint_epoch_{epoch:02d}.keras'),
            save_freq='epoch',
            save_weights_only=False,
            verbose=0,  # Reduced verbosity since we have live updates
            save_best_only=False  # Save every epoch
        )
        callbacks.append(checkpoint_callback)
        print(f"[OK] Checkpoint callback configured: will save after each epoch to {self.output_dir}")
        print(f"[OK] Live progress updates enabled - you'll see real-time training metrics!")
        sys.stdout.flush()
        
        # Prepare training data
        if self.use_metadata_features:
            train_data = [X_train, metadata_train]
            val_data = ([X_val, metadata_val], y_val) if y_val is not None else None
        else:
            train_data = X_train
            val_data = (X_val, y_val) if y_val is not None else None
        
        # Train
        self.history = self.model.fit(
            train_data, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val_data,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("[OK] Training completed")
        
        # Optimize threshold
        if self.optimize_threshold and y_val is not None:
            self.optimal_threshold = self.optimize_threshold(
                X_val, y_val, metadata_val if self.use_metadata_features else None
            )
        
        # Save model
        final_model_path = self.output_dir / 'final_model.keras'
        self.model.save(final_model_path)
        print(f"  Saved model: {final_model_path}")
        
        return self.history
    
    def evaluate(
        self,
        X_test,
        y_test: np.ndarray,
        metadata_test=None
    ) -> dict:
        """Evaluate model with per-class metrics."""
        print("\nEvaluating model...")
        
        # Predictions
        if self.use_metadata_features:
            test_data = [X_test, metadata_test]
        else:
            test_data = X_test
        
        y_pred_proba = self.model.predict(test_data, batch_size=self.batch_size, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        
        # Use optimal threshold
        threshold = self.optimal_threshold if self.optimize_threshold else 0.5
        y_pred = (y_pred_proba_flat > threshold).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_spam = f1_score(y_test, y_pred, pos_label=1, average='binary')
        f1_non_spam = f1_score(y_test, y_pred, pos_label=0, average='binary')
        precision_spam = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall_spam = recall_score(y_test, y_pred, pos_label=1, average='binary')
        precision_non_spam = precision_score(y_test, y_pred, pos_label=0, average='binary')
        recall_non_spam = recall_score(y_test, y_pred, pos_label=0, average='binary')
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba_flat)
        except:
            auc = None
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_spam': f1_spam,
            'f1_non_spam': f1_non_spam,
            'precision_spam': precision_spam,
            'recall_spam': recall_spam,
            'precision_non_spam': precision_non_spam,
            'recall_non_spam': recall_non_spam,
            'threshold': threshold,
            'roc_auc': auc
        }
        
        print(f"\n[OK] Test Accuracy: {accuracy:.4f}")
        print(f"[OK] F1 (Weighted): {f1_weighted:.4f}")
        print(f"[OK] F1 (Spam): {f1_spam:.4f}")
        print(f"[OK] F1 (Non-Spam): {f1_non_spam:.4f}")
        print(f"[OK] Precision (Spam): {precision_spam:.4f}")
        print(f"[OK] Recall (Spam): {recall_spam:.4f}")
        print(f"[OK] Precision (Non-Spam): {precision_non_spam:.4f}")
        print(f"[OK] Recall (Non-Spam): {recall_non_spam:.4f}")
        if auc:
            print(f"[OK] ROC-AUC: {auc:.4f}")
        print(f"[OK] Decision Threshold: {threshold:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save misclassified examples
        self._save_misclassified(X_test, y_test, y_pred, y_pred_proba_flat, metadata_test)
        
        # Save metrics
        metrics_path = self.output_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Advanced Model Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics.items():
                if value is not None:
                    f.write(f"{key}: {value:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"  Saved metrics: {metrics_path}")
        
        return metrics
    
    def _save_misclassified(self, X_test, y_test, y_pred, y_pred_proba, metadata_test=None):
        """Save misclassified examples for analysis."""
        misclassified = []
        
        # Get original texts (need to reverse tokenize)
        # Note: We'll save sequence indices since reverse tokenization requires full vocabulary
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                # Get sequence (first 50 non-zero tokens for readability)
                sequence = X_test[i]
                non_zero = sequence[sequence != 0][:50]
                
                misclassified.append({
                    'sequence_indices': str(non_zero.tolist()),
                    'true_label': 'Spam' if y_test[i] == 1 else 'Non-Spam',
                    'predicted_label': 'Spam' if y_pred[i] == 1 else 'Non-Spam',
                    'probability': float(y_pred_proba[i]),
                    'length_norm': float(metadata_test[i][0]) if metadata_test is not None else None,
                    'capital_ratio': float(metadata_test[i][1]) if metadata_test is not None else None,
                    'punct_density': float(metadata_test[i][2]) if metadata_test is not None else None,
                    'rating_norm': float(metadata_test[i][3]) if metadata_test is not None and len(metadata_test[i]) > 3 else None
                })
        
        if misclassified:
            misclassified_df = pd.DataFrame(misclassified)
            misclassified_path = self.output_dir / 'misclassified_examples.csv'
            misclassified_df.to_csv(misclassified_path, index=False)
            print(f"  Saved {len(misclassified)} misclassified examples: {misclassified_path}")
    
    def plot_precision_recall_curve(self, X_test, y_test: np.ndarray, metadata_test=None):
        """Plot precision-recall curve."""
        print("\nGenerating precision-recall curve...")
        
        if self.use_metadata_features:
            test_data = [X_test, metadata_test]
        else:
            test_data = X_test
        
        y_pred_proba = self.model.predict(test_data, batch_size=self.batch_size, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_flat)
        avg_precision = np.trapz(precision, recall)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.axvline(x=recall[np.argmax(precision + recall)], color='r', linestyle='--', 
                   label=f'Optimal Threshold: {self.optimal_threshold:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Spam Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pr_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {pr_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            return
        
        print("\nGenerating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        if 'val_precision' in self.history.history:
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        if 'val_recall' in self.history.history:
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_path = self.output_dir / 'training_history.png'
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {history_path}")
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=['Non-Spam', 'Spam'],
                   yticklabels=['Non-Spam', 'Spam'])
        plt.title('Confusion Matrix - Advanced Spam Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {cm_path}")
    
    def plot_roc_curve(self, X_test, y_test: np.ndarray, metadata_test=None):
        """Plot ROC curve."""
        print("\nGenerating ROC curve...")
        
        if self.use_metadata_features:
            test_data = [X_test, metadata_test]
        else:
            test_data = X_test
        
        y_pred_proba = self.model.predict(test_data, batch_size=self.batch_size, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_flat)
        auc = roc_auc_score(y_test, y_pred_proba_flat)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Advanced Spam Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = self.output_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {roc_path}")
    
    def run_full_pipeline(self):
        """Run complete pipeline."""
        print("=" * 70)
        print("Advanced RNN Classification Pipeline")
        print("=" * 70)
        
        # Load data
        texts, y_labels, y_ratings = self.load_data()
        
        # Preprocess texts
        X_sequences = self.preprocess_texts(texts)
        
        # Extract metadata features
        if self.use_metadata_features:
            metadata_features = self._extract_metadata_features(texts, y_ratings.tolist() if y_ratings is not None else None)
        else:
            metadata_features = None
        
        # Load GloVe embeddings if enabled
        if self.use_pretrained_embeddings:
            glove_file = self.download_glove_embeddings()
            if glove_file:
                embeddings_index = self.load_glove_embeddings(glove_file)
                self.embedding_matrix = self.create_embedding_matrix(
                    self.tokenizer.word_index,
                    embeddings_index
                )
            else:
                self.embedding_matrix = None
                print("  Using random embeddings (GloVe download failed)")
        else:
            self.embedding_matrix = None
        
        # Split data
        if self.use_metadata_features:
            X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                X_sequences, y_labels, metadata_features,
                test_size=0.2,
                random_state=42,
                stratify=y_labels
            )
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                X_train, y_train, meta_train,
                test_size=self.validation_split,
                random_state=42,
                stratify=y_train
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_labels,
                test_size=0.2,
                random_state=42,
                stratify=y_labels
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.validation_split,
                random_state=42,
                stratify=y_train
            )
            meta_train = meta_val = meta_test = None
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train):,}")
        print(f"  Validation: {len(X_val):,}")
        print(f"  Test: {len(X_test):,}")
        
        # Build model
        metadata_dim = 4 if self.use_metadata_features else 0
        self.build_model(num_classes=1, metadata_dim=metadata_dim)
        
        # Train
        self.train(
            X_train, y_train,
            X_val, y_val,
            meta_train, meta_val
        )
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test, meta_test)
        
        # Predictions for plots
        if self.use_metadata_features:
            test_data = [X_test, meta_test]
        else:
            test_data = X_test
        
        y_pred_proba = self.model.predict(test_data, batch_size=self.batch_size, verbose=0)
        y_pred_proba_flat = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        y_pred = (y_pred_proba_flat > self.optimal_threshold).astype(int)
        
        # Generate plots
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(X_test, y_test, meta_test)
        self.plot_precision_recall_curve(X_test, y_test, meta_test)
        
        print("\n" + "=" * 70)
        print("Pipeline Complete!")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"\nFinal Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (Spam): {metrics['f1_spam']:.4f} (Target: >0.90)")
        print(f"  F1 (Non-Spam): {metrics['f1_non_spam']:.4f} (Target: >0.93)")
        print(f"  Optimal Threshold: {metrics['threshold']:.3f}")


def main():
    """Main execution function."""
    # ==================== CONFIGURATION ====================
    INPUT_FILE = "output/balanced_rf/balanced_dataset.csv"  # Using balanced dataset (50/50 spam/non-spam)
    SAMPLE_SIZE = None  # None = use all data, or set limit for testing
    
    # Feature flags
    USE_PRETRAINED_EMBEDDINGS = True  # Use GloVe 200d embeddings
    USE_ATTENTION = True  # Use attention mechanism
    USE_METADATA_FEATURES = True  # Use metadata features
    OPTIMIZE_THRESHOLD = True  # Optimize decision threshold
    
    # Hyperparameters - OPTIMIZED FOR YOUR HARDWARE
    MAX_WORDS = 20000
    MAX_LENGTH = 150
    BATCH_SIZE = 256  # INCREASED from 128 (you have 31GB RAM)
    EPOCHS = 20
    # =======================================================
    
    print("=" * 70)
    print("Advanced RNN Spam Classifier")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL DATA'}")
    print(f"  Max words: {MAX_WORDS}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"\nFeature Flags:")
    print(f"  Pre-trained embeddings: {USE_PRETRAINED_EMBEDDINGS}")
    print(f"  Attention mechanism: {USE_ATTENTION}")
    print(f"  Metadata features: {USE_METADATA_FEATURES}")
    print(f"  Optimize threshold: {OPTIMIZE_THRESHOLD}")
    print("\nStarting pipeline...\n")
    
    try:
        classifier = AdvancedSpamClassifier(
            input_file=INPUT_FILE,
            max_words=MAX_WORDS,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            sample_size=SAMPLE_SIZE,
            use_pretrained_embeddings=USE_PRETRAINED_EMBEDDINGS,
            use_attention=USE_ATTENTION,
            use_metadata_features=USE_METADATA_FEATURES,
            optimize_threshold=OPTIMIZE_THRESHOLD
        )
        
        classifier.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("SUCCESS! Training completed.")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] ERROR: File not found: {e}")
        print("   Make sure 'cleaned_dataset.csv' is in the current directory.")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

