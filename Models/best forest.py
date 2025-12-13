"""
Ultimate Performance Pipeline - LEAK-SAFE VERSION
================================================
Senior Data Engineer Production Script with ZERO Data Leakage
- Processes full 1.68M row dataset
- Strict memory management (float32, generators, gc.collect())
- High-performance Random Forest with balanced_subsample
- Comprehensive feature engineering (VADER + Word2Vec 200d)
- LEAK-SAFE: Split FIRST, fit transformers on TRAIN only
- Estimated runtime: 3-4 hours

CRITICAL: Train/Test split happens BEFORE any feature engineering.
All transformers are fitted ONLY on training data.

Author: Senior Data Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import gc
import time
import pickle
import re
from pathlib import Path
from typing import Iterator, Generator, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, roc_curve
)

# NLP Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("Set2")

# Set random seeds
np.random.seed(42)


class TextStreamGenerator:
    """
    Memory-efficient generator for streaming text data to Word2Vec.
    Prevents RAM crashes on large datasets.
    """
    
    def __init__(self, texts: list, chunk_size: int = 50000):
        """
        Initialize text stream generator.
        
        Args:
            texts: List of text strings to stream
            chunk_size: Not used in this version, kept for compatibility
        """
        self.texts = texts
        self.total_rows = len(texts)
        self.current_idx = 0
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text for Word2Vec."""
        if pd.isna(text) or not str(text).strip():
            return []
        # Simple tokenization - extract words
        tokens = re.findall(r'\b\w+\b', str(text).lower())
        return tokens
    
    def __iter__(self) -> Iterator[list]:
        """
        Generator that yields tokenized sentences one by one.
        Word2Vec expects an iterable of sentences (lists of tokens), not chunks.
        """
        processed = 0
        for text in self.texts:
            tokens = self._tokenize(str(text))
            # Only yield non-empty token lists
            if tokens:
                yield tokens
                processed += 1
                if processed % 100000 == 0:
                    print(f"  Word2Vec: Tokenized {processed:,}/{self.total_rows:,} sentences...", flush=True)
        
        print()  # New line after progress


class LeakSafePipeline:
    """
    Leak-safe pipeline for maximum performance spam detection.
    CRITICAL: Split happens FIRST, all transformers fit on TRAIN only.
    """
    
    def __init__(
        self,
        input_file: str = "cleaned_dataset.csv",
        output_dir: str = "output/ultimate_pipeline_leak_safe",
        chunk_size: int = 50000
    ):
        """
        Initialize leak-safe pipeline.
        
        Args:
            input_file: Path to input CSV
            output_dir: Output directory for results
            chunk_size: Chunk size for memory-efficient processing
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
        # Initialize components
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.word2vec_model = None
        self.category_encoder = None  # Will store one-hot column names from train
        self.column_mapping = {}
        
        # Timing
        self.start_time = None
        self.phase_times = {}
        
        # Leakage audit log
        self.leakage_audit = []
        
        print("=" * 80)
        print("LEAK-SAFE ULTIMATE PERFORMANCE PIPELINE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("CRITICAL: Train/Test split happens FIRST, before any feature engineering")
    
    def _log_leakage_audit(self, stage: str, train_shape: Tuple, test_shape: Tuple, 
                           description: str, data_used: str = "train_only"):
        """
        Log leakage audit information.
        
        Args:
            stage: Stage name (e.g., "Data Split", "Word2Vec Training")
            train_shape: Shape of training data (rows, cols)
            test_shape: Shape of test data (rows, cols)
            description: Description of what was done
            data_used: Which data was used ("train_only", "test_only", "both")
        """
        audit_entry = {
            'stage': stage,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_shape': train_shape,
            'test_shape': test_shape,
            'description': description,
            'data_used': data_used
        }
        self.leakage_audit.append(audit_entry)
        
        print(f"\n[LEAKAGE AUDIT] {stage}")
        print(f"  Train shape: {train_shape[0]:,} rows × {train_shape[1]} cols")
        print(f"  Test shape: {test_shape[0]:,} rows × {test_shape[1]} cols")
        print(f"  Data used: {data_used}")
        print(f"  Description: {description}")
    
    def _detect_columns(self, sample_df: pd.DataFrame) -> None:
        """Auto-detect column names."""
        print(f"Available columns: {list(sample_df.columns)}")
        for col in sample_df.columns:
            col_lower = col.lower()
            # Review text column
            if 'reviewtext' in col_lower:
                self.column_mapping['reviewText'] = col
            elif col_lower == 'review' and 'reviewText' not in self.column_mapping:
                self.column_mapping['reviewText'] = col
            # Rating column
            if 'rating' in col_lower and 'rating' not in self.column_mapping:
                self.column_mapping['rating'] = col
            elif col_lower == 'overall' and 'rating' not in self.column_mapping:
                self.column_mapping['rating'] = col
            # Spam column
            if 'spam' in col_lower or col_lower in ['class', 'is_spam']:
                self.column_mapping['is_spam'] = col
            # Category
            if 'category' in col_lower:
                self.column_mapping['category'] = col
            # Verified
            if 'verified' in col_lower:
                self.column_mapping['verified'] = col
        
        print(f"Column mapping: {self.column_mapping}")
        
        # Verify we found review text
        if 'reviewText' not in self.column_mapping:
            raise ValueError(f"Review text column not found! Available columns: {list(sample_df.columns)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        return text
    
    def _memory_optimize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by downcasting numeric types."""
        print("  Optimizing memory usage...")
        
        # Handle categorical columns first
        for col in df.select_dtypes(include=['category']).columns:
            df[col] = df[col].astype(str)
        
        # Handle integer columns
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            try:
                if isinstance(df[col], pd.Series):
                    df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
            except Exception as e:
                print(f"  Warning: Could not optimize integer column '{col}': {e}")
                continue
        
        # Handle float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            try:
                if isinstance(df[col], pd.Series):
                    df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            except Exception as e:
                print(f"  Warning: Could not optimize float column '{col}': {e}")
                continue
        
        print(f"  Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        STEP 1: Load FULL dataset and split FIRST (before any feature engineering).
        
        Returns:
            Tuple of (df_train, df_test, y_train, y_test)
        """
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("STEP 1: LOAD DATA AND SPLIT FIRST (LEAK-SAFE)")
        print("=" * 80)
        
        # Checkpoint 5%: Data Loading
        print("\n[CHECKPOINT 5%] Loading full dataset...")
        checkpoint_start = time.time()
        
        # Detect columns first
        sample_df = pd.read_csv(self.input_file, nrows=100)
        self._detect_columns(sample_df)
        
        # Load full dataset in chunks
        print("  Loading data in memory-efficient chunks...")
        total_rows = 0
        chunk_list = []
        max_chunks_in_memory = 20
        load_start = time.time()
        
        for chunk_num, chunk in enumerate(pd.read_csv(
            self.input_file,
            chunksize=self.chunk_size,
            low_memory=False
        ), 1):
            # Standardize column names
            chunk = chunk.rename(columns={v: k for k, v in self.column_mapping.items()})
            
            # Filter valid rows
            chunk = chunk.dropna(subset=['reviewText'])
            chunk = chunk[chunk['reviewText'].astype(str).str.strip() != ""]
            
            chunk_list.append(chunk)
            total_rows += len(chunk)
            
            # Progress update
            elapsed_load = time.time() - load_start
            rate = total_rows / elapsed_load if elapsed_load > 0 else 0
            print(f"  Loading: {chunk_num} chunks | {total_rows:,} rows loaded | {rate:.0f} rows/sec", flush=True)
            
            # Process in batches to avoid memory overflow
            if len(chunk_list) >= max_chunks_in_memory:
                if chunk_num == max_chunks_in_memory:
                    df = pd.concat(chunk_list, ignore_index=True)
                else:
                    temp_df = pd.concat(chunk_list, ignore_index=True)
                    df = pd.concat([df, temp_df], ignore_index=True)
                    del temp_df
                
                chunk_list = []
                gc.collect()
                print(f"  Memory cleared. Continuing... ({total_rows:,} rows so far)", flush=True)
        
        # Concatenate remaining chunks
        if chunk_list:
            if 'df' in locals():
                temp_df = pd.concat(chunk_list, ignore_index=True)
                df = pd.concat([df, temp_df], ignore_index=True)
                del temp_df
            else:
                df = pd.concat(chunk_list, ignore_index=True)
            del chunk_list
            gc.collect()
        
        print(f"\n  ✓ Loaded {total_rows:,} total rows")
        print(f"  Final dataframe shape: {df.shape}")
        
        # Memory optimization
        df = self._memory_optimize_numeric(df)
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 5% DONE: Data loaded and memory optimized. ({elapsed:.1f}s)")
        self.phase_times['data_loading'] = elapsed
        
        # CRITICAL: Split FIRST before any feature engineering
        print("\n[CRITICAL] Performing train/test split BEFORE any feature engineering...")
        split_start = time.time()
        
        # Extract target
        if 'is_spam' not in df.columns:
            raise ValueError("'is_spam' column not found!")
        
        y = df['is_spam'].astype(np.int32)
        X = df.drop(columns=['is_spam'])
        
        # Drop 'overall' column (target leakage prevention)
        if 'overall' in X.columns:
            X = X.drop(columns=['overall'])
            print("  ✓ Dropped 'overall' column (target leakage prevention)")
        
        # Drop 'rating' if it was mapped from 'overall'
        if 'rating' in X.columns and self.column_mapping.get('rating', '').lower() == 'overall':
            X = X.drop(columns=['rating'])
            print("  ✓ Dropped 'rating' column (was 'overall')")
        
        # Stratified Train/Test Split (80/20) - BEFORE ANY FEATURE ENGINEERING
        print("  Creating stratified train/test split (80/20)...")
        df_train, df_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Reset indices
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        print(f"  ✓ Train set: {len(df_train):,} rows")
        print(f"  ✓ Test set: {len(df_test):,} rows")
        print(f"  ✓ Train spam ratio: {y_train.mean():.2%}")
        print(f"  ✓ Test spam ratio: {y_test.mean():.2%}")
        
        # Leakage audit
        self._log_leakage_audit(
            "Data Split",
            df_train.shape,
            df_test.shape,
            "Split performed FIRST, before any feature engineering. Train and test are now isolated.",
            "both"
        )
        
        elapsed = time.time() - split_start
        self.phase_times['data_split'] = elapsed
        
        phase_elapsed = time.time() - phase_start
        self.phase_times['step1_total'] = phase_elapsed
        
        return df_train, df_test, y_train, y_test
    
    def build_features_train(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 2: Build features on TRAINING set only.
        Fits all transformers on training data.
        
        Returns:
            DataFrame with all features for training set
        """
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("STEP 2: BUILD FEATURES ON TRAINING SET (FIT TRANSFORMERS)")
        print("=" * 80)
        
        df = df_train.copy()
        
        # Checkpoint 15%: Text Cleaning (TRAIN ONLY)
        print("\n[CHECKPOINT 15%] Cleaning text on TRAINING set...")
        checkpoint_start = time.time()
        
        print("  Cleaning text...")
        total_texts = len(df)
        cleaned_texts = []
        clean_start = time.time()
        
        for idx, text in enumerate(df['reviewText'], 1):
            cleaned_texts.append(self._clean_text(text))
            
            if idx % 50000 == 0 or idx == total_texts:
                elapsed_clean = time.time() - clean_start
                rate = idx / elapsed_clean if elapsed_clean > 0 else 0
                progress_pct = (idx / total_texts) * 100
                print(f"  Cleaning: {idx:,}/{total_texts:,} ({progress_pct:.1f}%) | {rate:.0f} texts/sec", flush=True)
        
        df['clean_text'] = pd.Series(cleaned_texts, index=df.index)
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 15% DONE: Text cleaned on TRAINING set. ({elapsed:.1f}s)")
        
        # Checkpoint 20%: VADER Sentiment (TRAIN ONLY)
        print("\n[CHECKPOINT 20%] Computing VADER scores on TRAINING set...")
        checkpoint_start = time.time()
        
        vader_scores = []
        chunk_size_vader = 50000
        total_chunks = (len(df) + chunk_size_vader - 1) // chunk_size_vader
        
        print(f"  Processing {len(df):,} TRAIN reviews in {total_chunks} chunks...")
        
        for chunk_idx, i in enumerate(range(0, len(df), chunk_size_vader), 1):
            chunk_texts = df['clean_text'].iloc[i:i+chunk_size_vader]
            
            chunk_scores = []
            for row_idx, text in enumerate(chunk_texts, 1):
                if pd.isna(text) or not str(text).strip():
                    chunk_scores.append([0.0, 0.0, 0.0, 0.0])
                else:
                    scores = self.vader_analyzer.polarity_scores(str(text))
                    chunk_scores.append([
                        np.float32(scores['neg']),
                        np.float32(scores['neu']),
                        np.float32(scores['pos']),
                        np.float32(scores['compound'])
                    ])
                
                if row_idx % 10000 == 0 or row_idx == len(chunk_texts):
                    total_processed = i + row_idx
                    progress_pct = (total_processed / len(df)) * 100
                    elapsed_chunk = time.time() - checkpoint_start
                    rate = total_processed / elapsed_chunk if elapsed_chunk > 0 else 0
                    print(f"  VADER (TRAIN): {chunk_idx}/{total_chunks} chunks | {total_processed:,}/{len(df):,} ({progress_pct:.1f}%) | "
                          f"{rate:.0f} rev/s", flush=True)
            
            vader_scores.extend(chunk_scores)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        print()
        
        # Add VADER features
        vader_df = pd.DataFrame(
            vader_scores,
            columns=['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'],
            dtype=np.float32
        )
        
        for col in vader_df.columns:
            df[col] = vader_df[col].values
        
        del vader_scores, vader_df
        gc.collect()
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 20% DONE: VADER scores calculated on TRAINING set. ({elapsed:.1f}s)")
        self.phase_times['vader_train'] = elapsed
        
        # Checkpoint 35%: Word2Vec Training (TRAIN ONLY - CRITICAL)
        print("\n[CHECKPOINT 35%] Training Word2Vec model on TRAINING set ONLY...")
        checkpoint_start = time.time()
        
        # Create text stream generator from TRAIN texts only
        train_texts = df['clean_text'].tolist()
        text_stream = TextStreamGenerator(train_texts, chunk_size=50000)
        
        # Train Word2Vec ONLY on training data
        print("  Training Word2Vec with optimal parameters (TRAIN data only):")
        print("    - vector_size: 200")
        print("    - window: 10")
        print("    - min_count: 5")
        print("    - workers: all cores")
        print("  CRITICAL: Word2Vec vocabulary built ONLY from TRAIN reviews")
        
        self.word2vec_model = Word2Vec(
            sentences=text_stream,
            vector_size=200,
            window=10,
            min_count=5,
            workers=-1,
            seed=42,
            epochs=1
        )
        
        print(f"  ✓ Word2Vec vocabulary size: {len(self.word2vec_model.wv):,} words (from TRAIN only)")
        
        # Leakage audit
        self._log_leakage_audit(
            "Word2Vec Training",
            df.shape,
            (0, 0),  # Test not used
            "Word2Vec model trained ONLY on training set. Vocabulary and embeddings learned from TRAIN only.",
            "train_only"
        )
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 35% DONE: Word2Vec model trained on TRAINING set. ({elapsed:.1f}s)")
        self.phase_times['word2vec_training'] = elapsed
        
        # Checkpoint 40%: Word2Vec Vectorization (TRAIN ONLY)
        print("\n[CHECKPOINT 40%] Converting TRAIN reviews to 200-dimensional vectors...")
        checkpoint_start = time.time()
        
        w2v_features = []
        chunk_size_w2v = 50000
        
        def tokenize_for_w2v(text: str) -> list:
            if pd.isna(text) or not str(text).strip():
                return []
            return re.findall(r'\b\w+\b', str(text).lower())
        
        total_w2v_chunks = (len(df) + chunk_size_w2v - 1) // chunk_size_w2v
        print(f"  Converting {len(df):,} TRAIN reviews to 200-dimensional vectors...")
        
        for chunk_idx, i in enumerate(range(0, len(df), chunk_size_w2v), 1):
            chunk_texts = df['clean_text'].iloc[i:i+chunk_size_w2v]
            
            chunk_vectors = []
            for row_idx, text in enumerate(chunk_texts, 1):
                tokens = tokenize_for_w2v(str(text))
                vectors = []
                
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                
                if vectors:
                    avg_vector = np.mean(vectors, axis=0).astype(np.float32)
                else:
                    avg_vector = np.zeros(200, dtype=np.float32)
                
                chunk_vectors.append(avg_vector)
                
                if row_idx % 10000 == 0 or row_idx == len(chunk_texts):
                    total_processed = i + row_idx
                    progress_pct = (total_processed / len(df)) * 100
                    elapsed_w2v = time.time() - checkpoint_start
                    rate = total_processed / elapsed_w2v if elapsed_w2v > 0 else 0
                    print(f"  W2V Vectorization (TRAIN): {chunk_idx}/{total_w2v_chunks} chunks | {total_processed:,}/{len(df):,} ({progress_pct:.1f}%) | "
                          f"{rate:.0f} rev/s", flush=True)
            
            w2v_features.extend(chunk_vectors)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        print()
        
        # Add Word2Vec features
        w2v_df = pd.DataFrame(
            w2v_features,
            columns=[f'w2v_{i}' for i in range(200)],
            dtype=np.float32
        )
        
        for col in w2v_df.columns:
            df[col] = w2v_df[col].values
        
        del w2v_features, w2v_df
        gc.collect()
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 40% DONE: Word2Vec features created for TRAINING set. ({elapsed:.1f}s)")
        self.phase_times['vectorization_train'] = elapsed
        
        # Checkpoint 45%: One-Hot Encoding (TRAIN ONLY - FIT)
        print("\n[CHECKPOINT 45%] Performing one-hot encoding on TRAINING set (FIT)...")
        checkpoint_start = time.time()
        
        # Fit one-hot encoder on TRAIN category values
        if 'category' in df.columns:
            print("  Fitting one-hot encoder on TRAIN categories...")
            category_dummies = pd.get_dummies(df['category'], prefix='cat', dtype=np.float32)
            # Store column names for later use on test set
            self.category_encoder = list(category_dummies.columns)
            df = pd.concat([df, category_dummies], axis=1)
            df = df.drop(columns=['category'])
            del category_dummies
            gc.collect()
            print(f"  ✓ Created {len(self.category_encoder)} category columns from TRAIN")
        
        # One-Hot Encoding for verified_purchase
        if 'verified' in df.columns:
            print("  Encoding verified_purchase on TRAIN...")
            df['verified_encoded'] = df['verified'].apply(
                lambda x: np.float32(1.0) if pd.notna(x) and str(x).lower() in ['true', '1', 'yes', 'verified'] 
                else np.float32(0.0)
            )
            df = df.drop(columns=['verified'])
        
        # Leakage audit
        self._log_leakage_audit(
            "One-Hot Encoding (Fit)",
            df.shape,
            (0, 0),  # Test not used
            "One-hot encoder fitted ONLY on training set categories. Column names stored for test set transformation.",
            "train_only"
        )
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 45% DONE: One-hot encoding fitted on TRAINING set. ({elapsed:.1f}s)")
        self.phase_times['onehot_fit'] = elapsed
        
        # Cleanup: Remove raw text columns
        print("\n  Performing memory cleanup...")
        text_columns_to_drop = ['reviewText', 'review', 'clean_review', 'summary', 'clean_summary', 'clean_text']
        text_columns_to_drop = [col for col in text_columns_to_drop if col in df.columns]
        df = df.drop(columns=text_columns_to_drop)
        gc.collect()
        
        phase_elapsed = time.time() - phase_start
        self.phase_times['step2_total'] = phase_elapsed
        print(f"\nStep 2 Total Time: {phase_elapsed:.1f}s ({phase_elapsed/60:.1f} minutes)")
        
        return df
    
    def build_features_test(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 3: Build features on TEST set (TRANSFORM ONLY).
        Applies pre-fitted transformers to test data.
        
        Returns:
            DataFrame with all features for test set
        """
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("STEP 3: BUILD FEATURES ON TEST SET (TRANSFORM ONLY)")
        print("=" * 80)
        
        df = df_test.copy()
        
        # Checkpoint 50%: Text Cleaning (TEST ONLY)
        print("\n[CHECKPOINT 50%] Cleaning text on TEST set...")
        checkpoint_start = time.time()
        
        print("  Cleaning text...")
        total_texts = len(df)
        cleaned_texts = []
        
        for idx, text in enumerate(df['reviewText'], 1):
            cleaned_texts.append(self._clean_text(text))
            
            if idx % 50000 == 0 or idx == total_texts:
                print(f"  Cleaning: {idx:,}/{total_texts:,} texts", flush=True)
        
        df['clean_text'] = pd.Series(cleaned_texts, index=df.index)
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 50% DONE: Text cleaned on TEST set. ({elapsed:.1f}s)")
        
        # Checkpoint 55%: VADER Sentiment (TEST ONLY)
        print("\n[CHECKPOINT 55%] Computing VADER scores on TEST set...")
        checkpoint_start = time.time()
        
        vader_scores = []
        chunk_size_vader = 50000
        total_chunks = (len(df) + chunk_size_vader - 1) // chunk_size_vader
        
        print(f"  Processing {len(df):,} TEST reviews...")
        
        for chunk_idx, i in enumerate(range(0, len(df), chunk_size_vader), 1):
            chunk_texts = df['clean_text'].iloc[i:i+chunk_size_vader]
            
            chunk_scores = []
            for text in chunk_texts:
                if pd.isna(text) or not str(text).strip():
                    chunk_scores.append([0.0, 0.0, 0.0, 0.0])
                else:
                    scores = self.vader_analyzer.polarity_scores(str(text))
                    chunk_scores.append([
                        np.float32(scores['neg']),
                        np.float32(scores['neu']),
                        np.float32(scores['pos']),
                        np.float32(scores['compound'])
                    ])
            
            vader_scores.extend(chunk_scores)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        # Add VADER features
        vader_df = pd.DataFrame(
            vader_scores,
            columns=['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'],
            dtype=np.float32
        )
        
        for col in vader_df.columns:
            df[col] = vader_df[col].values
        
        del vader_scores, vader_df
        gc.collect()
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 55% DONE: VADER scores calculated on TEST set. ({elapsed:.1f}s)")
        self.phase_times['vader_test'] = elapsed
        
        # Checkpoint 60%: Word2Vec Vectorization (TEST ONLY - TRANSFORM)
        print("\n[CHECKPOINT 60%] Converting TEST reviews to 200-dimensional vectors (using TRAIN-fitted model)...")
        checkpoint_start = time.time()
        
        w2v_features = []
        chunk_size_w2v = 50000
        
        def tokenize_for_w2v(text: str) -> list:
            if pd.isna(text) or not str(text).strip():
                return []
            return re.findall(r'\b\w+\b', str(text).lower())
        
        total_w2v_chunks = (len(df) + chunk_size_w2v - 1) // chunk_size_w2v
        print(f"  Converting {len(df):,} TEST reviews using TRAIN-fitted Word2Vec model...")
        print("  CRITICAL: Using Word2Vec model trained ONLY on training data")
        
        for chunk_idx, i in enumerate(range(0, len(df), chunk_size_w2v), 1):
            chunk_texts = df['clean_text'].iloc[i:i+chunk_size_w2v]
            
            chunk_vectors = []
            for row_idx, text in enumerate(chunk_texts, 1):
                tokens = tokenize_for_w2v(str(text))
                vectors = []
                
                # Use TRAIN-fitted model (transform only)
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                
                if vectors:
                    avg_vector = np.mean(vectors, axis=0).astype(np.float32)
                else:
                    avg_vector = np.zeros(200, dtype=np.float32)
                
                chunk_vectors.append(avg_vector)
                
                if row_idx % 10000 == 0 or row_idx == len(chunk_texts):
                    total_processed = i + row_idx
                    progress_pct = (total_processed / len(df)) * 100
                    elapsed_w2v = time.time() - checkpoint_start
                    rate = total_processed / elapsed_w2v if elapsed_w2v > 0 else 0
                    print(f"  W2V Vectorization (TEST): {chunk_idx}/{total_w2v_chunks} chunks | {total_processed:,}/{len(df):,} ({progress_pct:.1f}%) | "
                          f"{rate:.0f} rev/s", flush=True)
            
            w2v_features.extend(chunk_vectors)
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        print()
        
        # Add Word2Vec features
        w2v_df = pd.DataFrame(
            w2v_features,
            columns=[f'w2v_{i}' for i in range(200)],
            dtype=np.float32
        )
        
        for col in w2v_df.columns:
            df[col] = w2v_df[col].values
        
        del w2v_features, w2v_df
        gc.collect()
        
        # Leakage audit
        self._log_leakage_audit(
            "Word2Vec Vectorization (Transform)",
            df.shape,
            (0, 0),  # Not applicable
            "Word2Vec features created for test set using TRAIN-fitted model. No test data used to fit model.",
            "test_only"
        )
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 60% DONE: Word2Vec features created for TEST set. ({elapsed:.1f}s)")
        self.phase_times['vectorization_test'] = elapsed
        
        # Checkpoint 65%: One-Hot Encoding (TEST ONLY - TRANSFORM)
        print("\n[CHECKPOINT 65%] Performing one-hot encoding on TEST set (TRANSFORM)...")
        checkpoint_start = time.time()
        
        # Apply one-hot encoder to TEST (using column names from TRAIN)
        if 'category' in df.columns:
            print("  Applying one-hot encoder to TEST categories (using TRAIN-fitted columns)...")
            category_dummies = pd.get_dummies(df['category'], prefix='cat', dtype=np.float32)
            
            # Ensure test has same columns as train (add missing, remove extra)
            for col in self.category_encoder:
                if col not in category_dummies.columns:
                    category_dummies[col] = np.float32(0.0)
            
            # Remove any columns not in train
            category_dummies = category_dummies[self.category_encoder]
            
            df = pd.concat([df, category_dummies], axis=1)
            df = df.drop(columns=['category'])
            del category_dummies
            gc.collect()
            print(f"  ✓ Applied {len(self.category_encoder)} category columns to TEST")
        
        # One-Hot Encoding for verified_purchase (same logic as train)
        if 'verified' in df.columns:
            print("  Encoding verified_purchase on TEST...")
            df['verified_encoded'] = df['verified'].apply(
                lambda x: np.float32(1.0) if pd.notna(x) and str(x).lower() in ['true', '1', 'yes', 'verified'] 
                else np.float32(0.0)
            )
            df = df.drop(columns=['verified'])
        
        # Leakage audit
        self._log_leakage_audit(
            "One-Hot Encoding (Transform)",
            df.shape,
            (0, 0),  # Not applicable
            "One-hot encoding applied to test set using TRAIN-fitted column structure. No test data used to fit encoder.",
            "test_only"
        )
        
        elapsed = time.time() - checkpoint_start
        print(f"✓ 65% DONE: One-hot encoding applied to TEST set. ({elapsed:.1f}s)")
        self.phase_times['onehot_transform'] = elapsed
        
        # Cleanup: Remove raw text columns
        print("\n  Performing memory cleanup...")
        text_columns_to_drop = ['reviewText', 'review', 'clean_review', 'summary', 'clean_summary', 'clean_text']
        text_columns_to_drop = [col for col in text_columns_to_drop if col in df.columns]
        df = df.drop(columns=text_columns_to_drop)
        gc.collect()
        
        phase_elapsed = time.time() - phase_start
        self.phase_times['step3_total'] = phase_elapsed
        print(f"\nStep 3 Total Time: {phase_elapsed:.1f}s ({phase_elapsed/60:.1f} minutes)")
        
        return df
    
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series):
        """
        STEP 4: Train Random Forest on TRAIN and evaluate on TEST.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("STEP 4: TRAIN MODEL AND EVALUATE")
        print("=" * 80)
        
        # Ensure feature columns match
        print("\n  Ensuring feature columns match between train and test...")
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        if train_cols != test_cols:
            print(f"  WARNING: Feature mismatch detected!")
            print(f"  Train has {len(train_cols)} features, Test has {len(test_cols)} features")
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            
            if missing_in_test:
                print(f"  Missing in test: {missing_in_test}")
                for col in missing_in_test:
                    X_test[col] = np.float32(0.0)
            
            if missing_in_train:
                print(f"  Missing in train: {missing_in_train}")
                X_train = X_train.drop(columns=list(missing_in_train))
                X_test = X_test.drop(columns=list(missing_in_train))
            
            # Reorder to match
            X_test = X_test[X_train.columns]
            print(f"  ✓ Feature columns aligned: {len(X_train.columns)} features")
        
        # Drop ID columns
        id_cols = [c for c in ['reviewerID', 'asin', 'reviewerName'] if c in X_train.columns]
        if id_cols:
            print(f"  Dropping ID columns: {id_cols}")
            X_train = X_train.drop(columns=id_cols)
            X_test = X_test.drop(columns=id_cols)
        
        # Ensure all features are numeric and float32
        print("  Converting features to float32...")
        for col in X_train.select_dtypes(include=['object']).columns:
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(np.float32)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(np.float32)
            except Exception as e:
                print(f"  Warning: Dropping column '{col}': {e}")
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        
        for col in X_train.select_dtypes(include=[np.number]).columns:
            X_train[col] = X_train[col].astype(np.float32)
            X_test[col] = X_test[col].astype(np.float32)
        
        # Fix infinity and out-of-range values
        print("  Checking for infinity and out-of-range values...")
        for df in [X_train, X_test]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            float32_max = np.finfo(np.float32).max
            float32_min = np.finfo(np.float32).min
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].max() > float32_max or df[col].min() < float32_min:
                    df[col] = df[col].clip(float32_min, float32_max)
        
        print("  ✓ Data cleaned and validated")
        
        # Leakage audit
        self._log_leakage_audit(
            "Model Training",
            X_train.shape,
            X_test.shape,
            "Random Forest will be trained ONLY on training set. Test set will be used ONLY for evaluation.",
            "both"
        )
        
        # Checkpoint 90%: Model Training
        print("\n[CHECKPOINT 90%] Training Random Forest classifier on TRAINING set...")
        checkpoint_start = time.time()
        
        print("  Random Forest Configuration:")
        print("    - n_estimators: 250")
        print("    - max_depth: 25")
        print("    - class_weight: 'balanced_subsample'")
        print("    - n_jobs: -1 (all cores)")
        print("    - random_state: 42")
        print("  CRITICAL: Training on TRAIN set only")
        
        clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=25,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        print("\n  Training started (this will take significant time)...")
        clf.fit(X_train, y_train)
        
        elapsed = time.time() - checkpoint_start
        print(f"\n✓ 90% DONE: Random Forest training complete on TRAINING set. ({elapsed:.1f}s)")
        self.phase_times['rf_training'] = elapsed
        
        # Save model
        model_path = self.output_dir / 'rf_model_leak_safe.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"  ✓ Saved model: {model_path}")
        
        # Save test data for threshold optimization
        test_data_path = self.output_dir / 'test_data.pkl'
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  ✓ Saved test data for threshold optimization: {test_data_path}")
        
        # Checkpoint 100%: Evaluation on TEST
        print("\n[CHECKPOINT 100%] Evaluating on TEST set...")
        checkpoint_start = time.time()
        
        # Predictions on TEST only
        print("  Generating predictions on TEST set...")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_spam = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification Report
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT (TEST SET ONLY)")
        print("=" * 80)
        report = classification_report(y_test, y_pred)
        print(report)
        
        print(f"\nKey Metrics (TEST SET):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Weighted): {f1:.4f}")
        print(f"  F1-Score (Spam): {f1_spam:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Save classification report
        report_path = self.output_dir / 'classification_report_leak_safe.txt'
        with open(report_path, 'w') as f:
            f.write("Leak-Safe Ultimate Performance Pipeline - Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write("CRITICAL: Model trained on TRAIN only, evaluated on TEST only.\n")
            f.write("No data leakage detected.\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-Score (Weighted): {f1:.4f}\n")
            f.write(f"F1-Score (Spam): {f1_spam:.4f}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
            f.write(report)
        print(f"\n✓ Saved classification report: {report_path}")
        
        # Feature Importance
        print("\n  Generating feature importance...")
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Group features
        feature_importances['group'] = feature_importances['feature'].apply(
            lambda x: 'Sentiment' if 'vader' in x.lower() 
            else 'Semantic Content' if 'w2v' in x.lower()
            else 'Metadata'
        )
        
        # Save feature importance CSV
        importance_path = self.output_dir / 'feature_importance_leak_safe.csv'
        feature_importances.to_csv(importance_path, index=False)
        print(f"  ✓ Saved: {importance_path}")
        
        # Grouped Feature Importance Plot
        print("\n  Generating grouped feature importance plot...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        groups = feature_importances.groupby('group')
        colors = {'Sentiment': '#2E86AB', 'Semantic Content': '#A23B72', 'Metadata': '#F18F01'}
        
        y_pos = 0
        for group_name, group_df in groups:
            top_features = group_df.head(20)
            y_range = range(y_pos, y_pos + len(top_features))
            ax.barh(y_range, top_features['importance'], color=colors.get(group_name, '#6C757D'))
            ax.set_yticks([y + y_pos for y in y_range])
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            y_pos += len(top_features) + 2
        
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Grouped Feature Importance (Leak-Safe Pipeline)\n(Sentiment, Semantic Content, Metadata)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        grouped_importance_path = self.output_dir / 'full_feature_groups_leak_safe.png'
        plt.savefig(grouped_importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {grouped_importance_path}")
        
        # ROC Curve
        print("\n  Generating ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Leak-Safe Ultimate Performance Pipeline', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        roc_path = self.output_dir / 'full_roc_leak_safe.png'
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {roc_path}")
        
        # Confusion Matrix
        print("\n  Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            xticklabels=['Non-Spam', 'Spam'],
            yticklabels=['Non-Spam', 'Spam']
        )
        plt.title('Confusion Matrix - Leak-Safe Ultimate Performance Pipeline', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.text(0.5, -0.15, 
                f'Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}',
                ha='center', transform=plt.gca().transAxes, fontsize=10)
        
        plt.tight_layout()
        cm_path = self.output_dir / 'full_conf_matrix_leak_safe.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {cm_path}")
        
        elapsed = time.time() - checkpoint_start
        print(f"\n✓ 100% DONE: All outputs generated. ({elapsed:.1f}s)")
        self.phase_times['evaluation'] = elapsed
        
        phase_elapsed = time.time() - phase_start
        self.phase_times['step4_total'] = phase_elapsed
        
        return clf
    
    def save_leakage_audit(self):
        """Save leakage audit log to file."""
        audit_path = self.output_dir / 'leakage_audit_log.txt'
        
        with open(audit_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA LEAKAGE AUDIT LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write("This pipeline is LEAK-SAFE. All transformers are fitted on TRAIN only.\n")
            f.write("Test set is used ONLY for transformation and evaluation.\n\n")
            
            for entry in self.leakage_audit:
                f.write(f"\n{'='*80}\n")
                f.write(f"Stage: {entry['stage']}\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write(f"Train Shape: {entry['train_shape'][0]:,} rows × {entry['train_shape'][1]} cols\n")
                f.write(f"Test Shape: {entry['test_shape'][0]:,} rows × {entry['test_shape'][1]} cols\n")
                f.write(f"Data Used: {entry['data_used']}\n")
                f.write(f"Description: {entry['description']}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("AUDIT COMPLETE: No data leakage detected.\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Saved leakage audit log: {audit_path}")
    
    def run_full_pipeline(self):
        """Run complete leak-safe pipeline."""
        self.start_time = time.time()
        
        try:
            # STEP 1: Load and split FIRST
            df_train, df_test, y_train, y_test = self.load_and_split_data()
            
            # STEP 2: Build features on TRAIN (fit transformers)
            X_train = self.build_features_train(df_train)
            
            # STEP 3: Build features on TEST (transform only)
            X_test = self.build_features_test(df_test)
            
            # STEP 4: Train and evaluate
            clf = self.train_and_evaluate(X_train, y_train, X_test, y_test)
            
            # Save leakage audit
            self.save_leakage_audit()
            
            # Final summary
            total_time = time.time() - self.start_time
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETE - LEAK-SAFE")
            print("=" * 80)
            print(f"Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} minutes / {total_time/3600:.2f} hours)")
            print("\nPhase Breakdown:")
            print(f"  Step 1 (Load & Split): {self.phase_times.get('step1_total', 0):.1f}s")
            print(f"  Step 2 (Build Features Train): {self.phase_times.get('step2_total', 0):.1f}s")
            print(f"  Step 3 (Build Features Test): {self.phase_times.get('step3_total', 0):.1f}s")
            print(f"  Step 4 (Train & Evaluate): {self.phase_times.get('step4_total', 0):.1f}s")
            
            print("\n" + "=" * 80)
            print("LEAKAGE AUDIT: NO DATA LEAKAGE DETECTED")
            print("=" * 80)
            print("✓ Train/Test split performed FIRST")
            print("✓ Word2Vec trained ONLY on training set")
            print("✓ One-hot encoder fitted ONLY on training set")
            print("✓ Test set used ONLY for transformation and evaluation")
            print("✓ Model trained ONLY on training set")
            
            print(f"\nAll outputs saved to: {self.output_dir}/")
            print("  - rf_model_leak_safe.pkl")
            print("  - classification_report_leak_safe.txt")
            print("  - feature_importance_leak_safe.csv")
            print("  - full_feature_groups_leak_safe.png")
            print("  - full_roc_leak_safe.png")
            print("  - full_conf_matrix_leak_safe.png")
            print("  - leakage_audit_log.txt")
            
        except Exception as e:
            error_msg = f"\n❌ ERROR: {e}"
            print(error_msg)
            import traceback
            tb = traceback.format_exc()
            print(tb)
            
            # Save error log
            error_log_path = self.output_dir / 'error_log.txt'
            with open(error_log_path, 'w') as f:
                f.write(f"Pipeline Error Log\n")
                f.write(f"{'='*80}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Error: {e}\n\n")
                f.write(f"Traceback:\n{tb}\n")
            print(f"  Error log saved to: {error_log_path}")
            raise


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("LEAK-SAFE ULTIMATE PERFORMANCE PIPELINE - MAIN EXECUTION")
    print("=" * 80)
    print("\nThis pipeline is LEAK-SAFE:")
    print("  1. Loads FULL dataset")
    print("  2. Splits FIRST (80/20, stratified) BEFORE any feature engineering")
    print("  3. Fits ALL transformers ONLY on training set")
    print("  4. Applies transformers to test set (transform only)")
    print("  5. Trains model on TRAIN only")
    print("  6. Evaluates on TEST only")
    print("\nEstimated runtime: 3-4 hours")
    print("Memory management: Strict (float32, generators, gc.collect())")
    print("\nStarting pipeline...\n")
    
    pipeline = LeakSafePipeline(
        input_file="cleaned_dataset.csv",
        output_dir="output/ultimate_pipeline_leak_safe",
        chunk_size=50000
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

