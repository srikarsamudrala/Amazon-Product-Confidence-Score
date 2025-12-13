"""
Balanced Random Forest Classifier for Amazon Reviews
====================================================
Creates a balanced dataset (50/50 spam/non-spam) and trains a highly-balanced
Random Forest classifier to address severe class imbalance.

Key Features:
- Balanced dataset: 50,000 spam + 50,000 non-spam = 100,000 rows
- Drops 'overall' column
- Uses VADER sentiment, Word2Vec embeddings, numeric features, category encoding
- Random Forest with class_weight='balanced'
- Fast execution on 100K rows
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# NLP Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim.models import Word2Vec

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("Set2")


class BalancedRFClassifier:
    """
    Balanced Random Forest Classifier for spam detection.
    Handles dataset balancing, feature engineering, and model training.
    """
    
    def __init__(
        self,
        input_file: str = "cleaned_dataset.csv",
        output_dir: str = "output/balanced_rf",
        sample_size: int = 100000,  # Total rows: 50K spam + 50K non-spam
        drop_overall: bool = True
    ):
        """
        Initialize balanced RF classifier.
        
        Args:
            input_file: Path to CSV file
            output_dir: Directory for outputs
            sample_size: Total sample size (will be split 50/50)
            drop_overall: Whether to drop 'overall' column
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.drop_overall = drop_overall
        
        # Feature engineering components
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.word2vec_model = None
        self.category_encoder = LabelEncoder()
        self.column_mapping = {}
        
        print("=" * 70)
        print("Balanced Random Forest Classifier")
        print("=" * 70)
        print(f"  Input file: {input_file}")
        print(f"  Output directory: {output_dir}")
        print(f"  Sample size: {sample_size:,} (balanced 50/50)")
        print(f"  Drop 'overall' column: {drop_overall}")
    
    def _detect_columns(self, df: pd.DataFrame) -> None:
        """Auto-detect column names."""
        print(f"\nAvailable columns: {list(df.columns)}")
        for col in df.columns:
            col_lower = col.lower()
            # Review text column - handle both 'review' and 'reviewText'
            if 'reviewtext' in col_lower:
                self.column_mapping['reviewText'] = col
            elif col_lower == 'review' and 'reviewText' not in self.column_mapping:
                self.column_mapping['reviewText'] = col
            if 'rating' in col_lower or col_lower == 'overall':
                self.column_mapping['rating'] = col
            if 'asin' in col_lower:
                self.column_mapping['asin'] = col
            if 'spam' in col_lower or col_lower in ['class', 'is_spam']:
                self.column_mapping['is_spam'] = col
            if 'category' in col_lower:
                self.column_mapping['category'] = col
            if 'helpful' in col_lower:
                self.column_mapping['helpful'] = col
        
        print(f"Column mapping: {self.column_mapping}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        return text
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text for Word2Vec."""
        if not text:
            return []
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def create_balanced_dataset(self) -> pd.DataFrame:
        """
        Create balanced dataset with 50/50 spam/non-spam split.
        
        Returns:
            Balanced DataFrame with sample_size rows
        """
        print("\n" + "=" * 70)
        print("STEP 1: Creating Balanced Dataset")
        print("=" * 70)
        
        # Detect columns
        sample_df = pd.read_csv(self.input_file, nrows=100)
        self._detect_columns(sample_df)
        
        # Verify required columns
        required = ['reviewText', 'is_spam']
        missing = [c for c in required if c not in self.column_mapping]
        if missing:
            raise ValueError(f"Required columns not found: {missing}")
        
        # Load data in chunks
        print("\nLoading data...")
        spam_rows = []
        non_spam_rows = []
        chunk_size = 50000
        target_spam = self.sample_size // 2
        target_non_spam = self.sample_size // 2
        
        for chunk in pd.read_csv(
            self.input_file,
            chunksize=chunk_size,
            low_memory=False
        ):
            # Standardize column names
            chunk = chunk.rename(columns={v: k for k, v in self.column_mapping.items()})
            
            # Filter valid rows
            chunk = chunk.dropna(subset=['reviewText'])
            chunk = chunk[chunk['reviewText'].astype(str).str.strip() != ""]
            
            # Extract spam labels
            chunk['is_spam'] = chunk['is_spam'].apply(
                lambda x: 1 if pd.notna(x) and (int(x) == 1 or str(x).lower() in ['1', 'true', 'spam', 'yes']) else 0
            )
            
            # Split by class
            spam_chunk = chunk[chunk['is_spam'] == 1].copy()
            non_spam_chunk = chunk[chunk['is_spam'] == 0].copy()
            
            # Add to lists
            remaining_spam = target_spam - len(spam_rows)
            remaining_non_spam = target_non_spam - len(non_spam_rows)
            
            if remaining_spam > 0:
                spam_rows.extend(spam_chunk.head(remaining_spam).to_dict('records'))
            
            if remaining_non_spam > 0:
                non_spam_rows.extend(non_spam_chunk.head(remaining_non_spam).to_dict('records'))
            
            print(f"  Collected: {len(spam_rows):,} spam, {len(non_spam_rows):,} non-spam", end='\r')
            
            # Break if we have enough
            if len(spam_rows) >= target_spam and len(non_spam_rows) >= target_non_spam:
                break
        
        print(f"\n✓ Collected: {len(spam_rows):,} spam, {len(non_spam_rows):,} non-spam")
        
        # Combine and shuffle
        all_rows = spam_rows + non_spam_rows
        df = pd.DataFrame(all_rows)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Created balanced dataset: {len(df):,} rows")
        print(f"  Spam: {df['is_spam'].sum():,} ({df['is_spam'].mean()*100:.1f}%)")
        print(f"  Non-spam: {(df['is_spam']==0).sum():,} ({(df['is_spam']==0).mean()*100:.1f}%)")
        
        # Drop 'overall' column if requested
        if self.drop_overall:
            if 'overall' in df.columns:
                df = df.drop(columns=['overall'])
                print("✓ Dropped 'overall' column")
            # Also check if 'rating' column exists and was mapped from 'overall'
            if 'rating' in df.columns:
                original_rating_col = self.column_mapping.get('rating', '')
                if original_rating_col.lower() == 'overall':
                    df = df.drop(columns=['rating'])
                    print("✓ Dropped 'rating' column (was 'overall')")
        
        # Save balanced dataset
        balanced_path = self.output_dir / 'balanced_dataset.csv'
        df.to_csv(balanced_path, index=False)
        print(f"✓ Saved balanced dataset: {balanced_path}")
        
        return df
    
    def extract_vader_features(self, texts: pd.Series) -> pd.DataFrame:
        """Extract VADER sentiment features."""
        print("\nExtracting VADER sentiment features...")
        features = []
        for text in texts:
            if pd.isna(text) or not str(text).strip():
                features.append([0.0, 0.0, 0.0, 0.0])
            else:
                scores = self.vader_analyzer.polarity_scores(str(text))
                features.append([
                    scores['neg'],
                    scores['neu'],
                    scores['pos'],
                    scores['compound']
                ])
        
        vader_df = pd.DataFrame(
            features,
            columns=['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
        )
        print(f"✓ Extracted VADER features: {vader_df.shape}")
        return vader_df
    
    def extract_word2vec_features(self, texts: pd.Series, vector_size: int = 100) -> pd.DataFrame:
        """Extract Word2Vec embeddings."""
        print("\nTraining Word2Vec model and extracting features...")
        
        # Tokenize all texts (keep all, even empty ones)
        tokenized_texts = [self._tokenize(str(text)) for text in texts]
        
        # Filter out empty token lists for training (but keep track of indices)
        non_empty_tokenized = [tokens for tokens in tokenized_texts if tokens]
        
        if not non_empty_tokenized:
            print("⚠ No valid texts for Word2Vec")
            return pd.DataFrame(np.zeros((len(texts), vector_size)), 
                              columns=[f'w2v_{i}' for i in range(vector_size)])
        
        # Train Word2Vec model on non-empty texts only
        self.word2vec_model = Word2Vec(
            sentences=non_empty_tokenized,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            seed=42
        )
        
        print(f"✓ Trained Word2Vec model (vocab size: {len(self.word2vec_model.wv)} words)")
        
        # Extract features for each text (including empty ones)
        features = []
        for tokens in tokenized_texts:
            vectors = []
            for token in tokens:
                if token in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[token])
            
            if vectors:
                # Average word vectors
                avg_vector = np.mean(vectors, axis=0)
            else:
                avg_vector = np.zeros(vector_size)
            
            features.append(avg_vector)
        
        word2vec_df = pd.DataFrame(
            features,
            columns=[f'w2v_{i}' for i in range(vector_size)]
        )
        print(f"✓ Extracted Word2Vec features: {word2vec_df.shape}")
        return word2vec_df
    
    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric features (votes, review length, etc.)."""
        print("\nExtracting numeric features...")
        
        numeric_features = {}
        
        # Review length
        if 'reviewText' in df.columns:
            numeric_features['review_length'] = df['reviewText'].astype(str).str.len()
            numeric_features['review_word_count'] = df['reviewText'].astype(str).str.split().str.len()
        
        # Helpful votes
        if 'helpful' in df.columns:
            helpful = df['helpful'].astype(str)
            # Try to extract numbers from helpful column (format might be "[x, y]")
            def parse_helpful(val):
                try:
                    # Extract numbers from string like "[5, 10]"
                    nums = re.findall(r'\d+', str(val))
                    if len(nums) >= 2:
                        return int(nums[0]), int(nums[1])
                    elif len(nums) == 1:
                        return int(nums[0]), 0
                except:
                    pass
                return 0, 0
            
            helpful_parsed = helpful.apply(parse_helpful)
            numeric_features['helpful_votes'] = [x[0] for x in helpful_parsed]
            numeric_features['total_votes'] = [x[1] for x in helpful_parsed]
        else:
            numeric_features['helpful_votes'] = 0
            numeric_features['total_votes'] = 0
        
        # Rating (if still exists)
        if 'rating' in df.columns:
            numeric_features['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        
        numeric_df = pd.DataFrame(numeric_features)
        print(f"✓ Extracted numeric features: {numeric_df.shape}")
        return numeric_df
    
    def encode_category(self, categories: pd.Series) -> pd.Series:
        """Encode category labels."""
        print("\nEncoding categories...")
        
        # Fill NaN with 'unknown'
        categories_filled = categories.fillna('unknown').astype(str)
        
        # Fit and transform
        encoded = self.category_encoder.fit_transform(categories_filled)
        
        print(f"✓ Encoded {len(self.category_encoder.classes_)} categories")
        return pd.Series(encoded, name='category_encoded')
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create all features from balanced dataset.
        
        Returns:
            Feature matrix X and target vector y
        """
        print("\n" + "=" * 70)
        print("STEP 2: Feature Engineering")
        print("=" * 70)
        
        texts = df['reviewText'].apply(self._clean_text)
        
        # Extract features
        vader_features = self.extract_vader_features(texts)
        word2vec_features = self.extract_word2vec_features(texts, vector_size=100)
        numeric_features = self.extract_numeric_features(df)
        
        # Category encoding
        if 'category' in df.columns:
            category_encoded = self.encode_category(df['category'])
        else:
            category_encoded = pd.Series([0] * len(df), name='category_encoded')
        
        # Combine all features
        feature_list = [vader_features, word2vec_features, numeric_features]
        
        X = pd.concat(feature_list + [category_encoded], axis=1)
        
        # Target
        y = df['is_spam'].astype(int)
        
        print(f"\n✓ Feature engineering complete")
        print(f"  Total features: {X.shape[1]}")
        print(f"  Feature breakdown:")
        print(f"    - VADER: {vader_features.shape[1]}")
        print(f"    - Word2Vec: {word2vec_features.shape[1]}")
        print(f"    - Numeric: {numeric_features.shape[1]}")
        print(f"    - Category: 1")
        
        return X, y
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Train balanced Random Forest and evaluate.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print("\n" + "=" * 70)
        print("STEP 3: Training & Evaluation")
        print("=" * 70)
        
        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train):,} rows")
        print(f"    - Spam: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"    - Non-spam: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
        print(f"  Test: {len(X_test):,} rows")
        print(f"    - Spam: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        print(f"    - Non-spam: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)")
        
        # Train Random Forest with class_weight='balanced'
        print("\nTraining Random Forest classifier...")
        print("  - n_estimators: 150")
        print("  - max_depth: 25")
        print("  - class_weight: 'balanced'")
        
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        clf.fit(X_train, y_train)
        print("✓ Training complete")
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{'='*70}")
        print("MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Classification Report
        print(f"\n{'='*70}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*70}")
        report = classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam'])
        print(report)
        
        # Save classification report
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Balanced Random Forest Classifier - Classification Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-Score (weighted): {f1:.4f}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
            f.write(report)
        print(f"\n✓ Saved classification report: {report_path}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, y_test, y_pred)
        
        # Feature Importance (top 20)
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = self.output_dir / 'feature_importance.csv'
        feature_importances.to_csv(importance_path, index=False)
        print(f"✓ Saved feature importance: {importance_path}")
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return clf
    
    def plot_confusion_matrix(self, cm: np.ndarray, y_test: pd.Series, y_pred: np.ndarray):
        """Plot confusion matrix."""
        print("\nGenerating confusion matrix plot...")
        
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
        plt.title('Confusion Matrix - Balanced Random Forest Classifier', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add metrics text
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}',
                ha='center', transform=plt.gca().transAxes, fontsize=10)
        
        plt.tight_layout()
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix: {cm_path}")
    
    def run_full_pipeline(self):
        """Run complete pipeline: balance dataset, engineer features, train, evaluate."""
        # Step 1: Create balanced dataset
        df = self.create_balanced_dataset()
        
        # Step 2: Feature engineering
        X, y = self.create_features(df)
        
        # Step 3: Train and evaluate
        model = self.train_and_evaluate(X, y)
        
        # Save model
        import pickle
        model_path = self.output_dir / 'balanced_rf_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✓ Saved model: {model_path}")
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("  - balanced_dataset.csv (50/50 balanced dataset)")
        print("  - classification_report.txt")
        print("  - confusion_matrix.png")
        print("  - feature_importance.csv")
        print("  - balanced_rf_model.pkl")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Balanced Random Forest Classifier - Standalone Execution")
    print("=" * 70)
    
    # Configuration
    INPUT_FILE = "cleaned_dataset.csv"
    SAMPLE_SIZE = 100000  # 50K spam + 50K non-spam
    DROP_OVERALL = True
    
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Sample size: {SAMPLE_SIZE:,} (balanced 50/50)")
    print(f"  Drop 'overall' column: {DROP_OVERALL}")
    
    try:
        classifier = BalancedRFClassifier(
            input_file=INPUT_FILE,
            sample_size=SAMPLE_SIZE,
            drop_overall=DROP_OVERALL
        )
        
        classifier.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("SUCCESS! Training completed.")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found: {e}")
        print("   Make sure 'cleaned_dataset.csv' is in the current directory.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

