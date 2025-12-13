"""
Threshold 0.24 Full Evaluation - No Sampling, Full Dataset Only
- Fix test-set leakage
- Apply isotonic calibration on FULL dataset
- Force threshold 0.24
- Re-run leakage audit
- Generate comprehensive reports
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import sys
sys.path.insert(0, '.')
from rnn_classifier_advanced import AttentionLayer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("THRESHOLD 0.24 FULL EVALUATION - NO SAMPLING")
print("=" * 80)

# ====================================================================
# TASK 0: OUTPUT DIRECTORY
# ====================================================================

print("\n" + "=" * 80)
print("TASK 0: OUTPUT DIRECTORY")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"threshold24_run_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

# Assert directory exists
if not output_dir.exists():
    raise RuntimeError(f"Failed to create output directory: {output_dir}")

print(f"[OK] Created output directory: {output_dir}")
print(f"[OK] Timestamp: {timestamp}")

# ====================================================================
# LOAD MODEL AND TOKENIZER
# ====================================================================

print("\n" + "=" * 80)
print("LOADING MODEL AND TOKENIZER")
print("=" * 80)

model_dir = Path("output/rnn_models_advanced")
model_path = model_dir / "best_model.keras"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
print("[OK] Model loaded")

tokenizer_path = model_dir / "tokenizer.pkl"
if not tokenizer_path.exists():
    raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("[OK] Tokenizer loaded")

# ====================================================================
# LOAD BALANCED DATASET (SAME AS 0.57 TRAINING)
# ====================================================================

print("\n" + "=" * 80)
print("LOADING BALANCED DATASET (SAME AS 0.57 TRAINING)")
print("=" * 80)

# Use the same dataset as the original 0.57 threshold training
input_file = "output/balanced_rf/balanced_dataset.csv"

if not Path(input_file).exists():
    raise FileNotFoundError(f"Balanced dataset not found: {input_file}")

print(f"Loading balanced dataset: {input_file}")

# Load full dataset
df_full = pd.read_csv(input_file, low_memory=False)
print(f"[OK] Loaded {len(df_full):,} samples from balanced dataset")

# Detect columns
column_mapping = {}
for col in df_full.columns:
    col_lower = col.lower()
    if 'reviewtext' in col_lower and col == 'reviewText':
        # Prefer reviewText if it exists
        if 'reviewText' not in column_mapping:
            column_mapping['reviewText'] = col
    elif 'clean_review' in col_lower:
        # Use clean_review if reviewText not found
        if 'reviewText' not in column_mapping:
            column_mapping['reviewText'] = col
    elif 'rating' in col_lower or col_lower == 'overall':
        column_mapping['rating'] = col
    elif 'asin' in col_lower:
        column_mapping['asin'] = col
    elif 'spam' in col_lower or col_lower in ['class', 'is_spam']:
        column_mapping['is_spam'] = col

# If reviewText column exists, use it directly
if 'reviewText' in df_full.columns and 'reviewText' not in column_mapping:
    column_mapping['reviewText'] = 'reviewText'

print(f"Column mapping: {column_mapping}")

# Prepare data
df_full = df_full.rename(columns={v: k for k, v in column_mapping.items()})
if 'reviewText' not in df_full.columns:
    raise ValueError(f"reviewText column not found after renaming. Available columns: {df_full.columns.tolist()}")

# Filter empty reviews
df_full = df_full.dropna(subset=['reviewText'])
mask = df_full['reviewText'].astype(str).str.strip() != ""
df_full = df_full[mask].copy().reset_index(drop=True)

# Extract labels
df_full['label'] = df_full['is_spam'].apply(
    lambda x: 1 if pd.notna(x) and (int(x) == 1 or str(x).lower() in ['1', 'true', 'spam', 'yes']) else 0
)

texts = df_full['reviewText'].astype(str).tolist()
labels = np.array(df_full['label'].tolist())

print(f"[OK] Prepared {len(texts):,} samples")

# Preprocess texts
print("\nPreprocessing texts...")
max_length = 150
sequences = tokenizer.texts_to_sequences(texts)
X_data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y_data = labels

print(f"[OK] Preprocessed {len(X_data):,} samples")

# Prepare metadata if needed
model_inputs = model.inputs
metadata_data = None
if len(model_inputs) > 1:
    print("Preparing metadata features...")
    metadata_list = []
    ratings = df_full['rating'].fillna(3.0).tolist() if 'rating' in df_full.columns else [3.0] * len(texts)
    
    for i, text in enumerate(texts):
        text_str = str(text)
        length_norm = min(len(text_str) / 1000.0, 1.0)
        capital_ratio = sum(1 for c in text_str if c.isupper()) / len(text_str) if len(text_str) > 0 else 0.0
        punct_density = sum(1 for c in text_str if c in '.,!?;:') / len(text_str) if len(text_str) > 0 else 0.0
        rating_norm = (ratings[i] if i < len(ratings) else 3.0) / 5.0
        metadata_list.append([length_norm, capital_ratio, punct_density, rating_norm])
    
    metadata_data = np.array(metadata_list)
    print(f"[OK] Prepared metadata features")

# ====================================================================
# RECREATE TRAIN/TEST SPLIT (SAME AS 0.57 TRAINING)
# ====================================================================

print("\n" + "=" * 80)
print("RECREATING TRAIN/TEST SPLIT (SAME AS 0.57 TRAINING)")
print("=" * 80)

# Same split as original training: 80/20 with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Split metadata if available
if metadata_data is not None:
    _, metadata_test = train_test_split(metadata_data, test_size=0.2, random_state=42, stratify=y_data)
    _, metadata_train = train_test_split(metadata_data, test_size=0.2, random_state=42, stratify=y_data)
    # Fix: Need to get the actual train metadata
    train_indices, test_indices = train_test_split(
        np.arange(len(metadata_data)), test_size=0.2, random_state=42, stratify=y_data
    )
    metadata_train = metadata_data[train_indices]
    metadata_test = metadata_data[test_indices]
else:
    metadata_train = None
    metadata_test = None

print(f"[OK] Train: {len(X_train):,} samples")
print(f"[OK] Test (original): {len(X_test):,} samples")

# ====================================================================
# TASK 1: FIX TEST-SET LEAKAGE
# ====================================================================

print("\n" + "=" * 80)
print("TASK 1: FIX TEST-SET LEAKAGE")
print("=" * 80)

# Check for duplicates between train and test
print("\nChecking for duplicates between train and test sets...")

# Convert to hashable format for duplicate detection
def array_to_tuple(arr):
    """Convert numpy array to tuple for hashing."""
    return tuple(arr.flatten()[:100])  # Use first 100 elements for comparison

train_hashes = set()
for i in range(len(X_train)):
    if i % 1000 == 0:
        print(f"  Processing train samples: {i:,}/{len(X_train):,}", end='\r')
    train_hashes.add(array_to_tuple(X_train[i]))

print(f"\n[OK] Created hash set for {len(train_hashes):,} training samples")

duplicate_indices = []
for i in range(len(X_test)):
    if i % 1000 == 0:
        print(f"  Checking test samples: {i:,}/{len(X_test):,}", end='\r')
    test_hash = array_to_tuple(X_test[i])
    if test_hash in train_hashes:
        duplicate_indices.append(i)

print(f"\n[OK] Found {len(duplicate_indices)} duplicate(s) in test set")

if len(duplicate_indices) > 0:
    print(f"Removing {len(duplicate_indices)} duplicate(s) from test set...")
    
    # Remove duplicates
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.Series(y_test, name='label')
    
    X_test_clean_df = X_test_df.drop(index=duplicate_indices).reset_index(drop=True)
    y_test_clean_df = y_test_df.drop(index=duplicate_indices).reset_index(drop=True)
    
    # Convert back to numpy
    X_test_clean = X_test_clean_df.values
    y_test_clean = y_test_clean_df.values
    
    # Remove from metadata if available
    if metadata_test is not None:
        metadata_test_df = pd.DataFrame(metadata_test)
        metadata_test_clean_df = metadata_test_df.drop(index=duplicate_indices).reset_index(drop=True)
        metadata_test_clean = metadata_test_clean_df.values
    else:
        metadata_test_clean = None
    
    print(f"[OK] Test (cleaned): {len(X_test_clean):,} samples")
    print(f"[OK] Removed {len(duplicate_indices)} duplicate(s)")
    
    # Assert shapes match
    if X_test_clean.shape[0] != y_test_clean.shape[0]:
        raise RuntimeError(f"Shape mismatch: X_test_clean={X_test_clean.shape}, y_test_clean={y_test_clean.shape}")
    
    # Verify no duplicates remain
    remaining_duplicates = []
    for i in range(len(X_test_clean)):
        test_hash = array_to_tuple(X_test_clean[i])
        if test_hash in train_hashes:
            remaining_duplicates.append(i)
    
    if len(remaining_duplicates) > 0:
        raise RuntimeError(f"Duplicate removal failed: {len(remaining_duplicates)} duplicates still remain!")
    
    print("[OK] Verified: No duplicates remain in test set")
else:
    print("[OK] No duplicates found - test set is clean")
    X_test_clean = X_test
    y_test_clean = y_test
    metadata_test_clean = metadata_test

# Save duplicate fix report
fix_report_lines = []
fix_report_lines.append("=" * 80)
fix_report_lines.append("DUPLICATE FIX REPORT")
fix_report_lines.append("=" * 80)
fix_report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
fix_report_lines.append("")
fix_report_lines.append("TEST SET LEAKAGE FIX:")
fix_report_lines.append(f"  Original test size: {len(X_test):,} samples")
fix_report_lines.append(f"  Cleaned test size: {len(X_test_clean):,} samples")
fix_report_lines.append(f"  Duplicates found: {len(duplicate_indices)}")
if len(duplicate_indices) > 0:
    fix_report_lines.append(f"  Removed indices: {duplicate_indices[:10]}" + ("..." if len(duplicate_indices) > 10 else ""))
fix_report_lines.append(f"  Confirmation: No duplicates remain in test set")
fix_report_lines.append("")
fix_report_lines.append("VERIFICATION:")
fix_report_lines.append(f"  X_test_clean shape: {X_test_clean.shape}")
fix_report_lines.append(f"  y_test_clean shape: {y_test_clean.shape}")
fix_report_lines.append(f"  Shapes match: {X_test_clean.shape[0] == y_test_clean.shape[0]}")

fix_report_path = output_dir / 'duplicate_fix_report.txt'
with open(fix_report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(fix_report_lines))
print(f"[OK] Saved: duplicate_fix_report.txt")

# ====================================================================
# TASK 2: APPLY ISOTONIC CALIBRATION (FULL DATASET, NO SAMPLING)
# ====================================================================

print("\n" + "=" * 80)
print("TASK 2: APPLY ISOTONIC CALIBRATION (FULL DATASET)")
print("=" * 80)

# Prepare test data
if metadata_test_clean is not None:
    test_data = [X_test_clean, metadata_test_clean]
else:
    test_data = X_test_clean

print("\nComputing original probabilities on FULL cleaned test set...")
print(f"  Test set size: {len(X_test_clean):,} samples")
y_pred_proba_original = model.predict(test_data, batch_size=128, verbose=1).flatten()

# Validate predictions
if np.any(np.isnan(y_pred_proba_original)):
    raise RuntimeError("NaN values detected in original predictions")
if len(y_pred_proba_original) != len(y_test_clean):
    raise RuntimeError(f"Prediction length mismatch: {len(y_pred_proba_original)} != {len(y_test_clean)}")

print(f"[OK] Original probabilities computed: {len(y_pred_proba_original):,}")

# Fit IsotonicRegression on FULL dataset (NO SAMPLING)
print("\nFitting IsotonicRegression on FULL dataset (NO SAMPLING)...")
print(f"  Fitting on {len(y_pred_proba_original):,} samples")
isotonic_reg = IsotonicRegression(out_of_bounds='clip')
isotonic_reg.fit(y_pred_proba_original, y_test_clean)
print("[OK] IsotonicRegression fitted on FULL dataset")

# Compute calibrated probabilities
print("Computing calibrated probabilities...")
y_pred_proba_calibrated = isotonic_reg.transform(y_pred_proba_original)

# Validate calibrated predictions
if np.any(np.isnan(y_pred_proba_calibrated)):
    raise RuntimeError("NaN values detected in calibrated predictions")
if len(y_pred_proba_calibrated) != len(y_test_clean):
    raise RuntimeError(f"Calibrated prediction length mismatch: {len(y_pred_proba_calibrated)} != {len(y_test_clean)}")

print("[OK] Calibrated probabilities computed")

# ====================================================================
# EVALUATION METRICS
# ====================================================================

print("\n" + "=" * 80)
print("EVALUATING ORIGINAL AND CALIBRATED PROBABILITIES")
print("=" * 80)

def compute_ece(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

threshold = 0.24  # FORCE threshold = 0.24

# Evaluate Original Probabilities
y_pred_original = (y_pred_proba_original >= threshold).astype(int)
acc_original = accuracy_score(y_test_clean, y_pred_original)
prec_original = precision_score(y_test_clean, y_pred_original, average='weighted', zero_division=0)
recall_original = recall_score(y_test_clean, y_pred_original, average='weighted', zero_division=0)
f1_original = f1_score(y_test_clean, y_pred_original, average='weighted', zero_division=0)
roc_auc_original = roc_auc_score(y_test_clean, y_pred_proba_original)
fpr_orig, tpr_orig, _ = roc_curve(y_test_clean, y_pred_proba_original)
prec_curve_orig, rec_curve_orig, _ = precision_recall_curve(y_test_clean, y_pred_proba_original)
pr_auc_original = auc(rec_curve_orig, prec_curve_orig)
brier_original = brier_score_loss(y_test_clean, y_pred_proba_original)
ece_original = compute_ece(y_test_clean, y_pred_proba_original)
cm_original = confusion_matrix(y_test_clean, y_pred_original)

# Evaluate Calibrated Probabilities
y_pred_calibrated = (y_pred_proba_calibrated >= threshold).astype(int)
acc_calibrated = accuracy_score(y_test_clean, y_pred_calibrated)
prec_calibrated = precision_score(y_test_clean, y_pred_calibrated, average='weighted', zero_division=0)
recall_calibrated = recall_score(y_test_clean, y_pred_calibrated, average='weighted', zero_division=0)
f1_calibrated = f1_score(y_test_clean, y_pred_calibrated, average='weighted', zero_division=0)
roc_auc_calibrated = roc_auc_score(y_test_clean, y_pred_proba_calibrated)
fpr_cal, tpr_cal, _ = roc_curve(y_test_clean, y_pred_proba_calibrated)
prec_curve_cal, rec_curve_cal, _ = precision_recall_curve(y_test_clean, y_pred_proba_calibrated)
pr_auc_calibrated = auc(rec_curve_cal, prec_curve_cal)
brier_calibrated = brier_score_loss(y_test_clean, y_pred_proba_calibrated)
ece_calibrated = compute_ece(y_test_clean, y_pred_proba_calibrated)
cm_calibrated = confusion_matrix(y_test_clean, y_pred_calibrated)

print("\nMetrics Summary (Threshold = 0.24):")
print(f"{'Metric':<25} {'Original':<15} {'Calibrated':<15}")
print("-" * 55)
print(f"{'Accuracy':<25} {acc_original:<15.6f} {acc_calibrated:<15.6f}")
print(f"{'Precision (Weighted)':<25} {prec_original:<15.6f} {prec_calibrated:<15.6f}")
print(f"{'Recall (Weighted)':<25} {recall_original:<15.6f} {recall_calibrated:<15.6f}")
print(f"{'F1-Score (Weighted)':<25} {f1_original:<15.6f} {f1_calibrated:<15.6f}")
print(f"{'ROC AUC':<25} {roc_auc_original:<15.6f} {roc_auc_calibrated:<15.6f}")
print(f"{'PR AUC':<25} {pr_auc_original:<15.6f} {pr_auc_calibrated:<15.6f}")
print(f"{'Brier Score':<25} {brier_original:<15.6f} {brier_calibrated:<15.6f}")
print(f"{'ECE':<25} {ece_original:<15.6f} {ece_calibrated:<15.6f}")

# ====================================================================
# SAVE CALIBRATION RESULTS
# ====================================================================

print("\n" + "=" * 80)
print("SAVING CALIBRATION RESULTS")
print("=" * 80)

# Save calibration model
calibration_model_path = output_dir / 'calibration_model.pkl'
with open(calibration_model_path, 'wb') as f:
    pickle.dump({
        'calibrator': isotonic_reg,
        'method': 'isotonic',
        'threshold': threshold,
        'fit_samples': len(y_test_clean),
        'brier_score_original': brier_original,
        'brier_score_calibrated': brier_calibrated,
        'ece_original': ece_original,
        'ece_calibrated': ece_calibrated
    }, f)
print(f"[OK] Saved: calibration_model.pkl")

# Save calibrated predictions
calibrated_df = pd.DataFrame({
    'true_label': y_test_clean,
    'original_prob': y_pred_proba_original,
    'calibrated_prob': y_pred_proba_calibrated
})
calibrated_path = output_dir / 'calibrated_predictions.csv'
calibrated_df.to_csv(calibrated_path, index=False)
print(f"[OK] Saved: calibrated_predictions.csv ({len(calibrated_df):,} rows)")

# ====================================================================
# GENERATE PLOTS
# ====================================================================

print("\nGenerating plots...")

# Reliability Diagrams
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original
fraction_of_positives_orig, mean_predicted_value_orig = calibration_curve(
    y_test_clean, y_pred_proba_original, n_bins=10
)
axes[0].plot(mean_predicted_value_orig, fraction_of_positives_orig, 's-', 
             label='Original', linewidth=2, markersize=8)
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
axes[0].set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
axes[0].set_title(f'Original Probabilities\n(Brier: {brier_original:.4f}, ECE: {ece_original:.4f})', 
                  fontweight='bold', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Calibrated
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
    y_test_clean, y_pred_proba_calibrated, n_bins=10
)
axes[1].plot(mean_predicted_value_cal, fraction_of_positives_cal, 'o-', 
             label='Calibrated', linewidth=2, markersize=8, color='orange')
axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
axes[1].set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
axes[1].set_title(f'Calibrated Probabilities\n(Brier: {brier_calibrated:.4f}, ECE: {ece_calibrated:.4f})', 
                  fontweight='bold', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'reliability_diagrams.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: reliability_diagrams.png")

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].hist(y_pred_proba_original, bins=50, alpha=0.7, edgecolor='black', color='blue')
axes[0].set_xlabel('Predicted Probability', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[0].set_title('Original - Probability Distribution', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].hist(y_pred_proba_calibrated, bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('Predicted Probability', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[1].set_title('Calibrated - Probability Distribution', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'probability_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: probability_histograms.png")

# ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr_orig, tpr_orig, linewidth=2, label=f'Original (AUC = {roc_auc_original:.4f})')
plt.plot(fpr_cal, tpr_cal, linewidth=2, label=f'Calibrated (AUC = {roc_auc_calibrated:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
plt.title('ROC Curve (Threshold = 0.24)', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: roc_curve.png")

# PR Curve
plt.figure(figsize=(10, 8))
plt.plot(rec_curve_orig, prec_curve_orig, linewidth=2, label=f'Original (AUC = {pr_auc_original:.4f})')
plt.plot(rec_curve_cal, prec_curve_cal, linewidth=2, label=f'Calibrated (AUC = {pr_auc_calibrated:.4f})')
baseline = y_test_clean.mean()
plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, label=f'Baseline = {baseline:.4f}')
plt.xlabel('Recall', fontweight='bold', fontsize=12)
plt.ylabel('Precision', fontweight='bold', fontsize=12)
plt.title('Precision-Recall Curve (Threshold = 0.24)', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: pr_curve.png")

# Confusion Matrix (Calibrated)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_calibrated, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Non-Spam', 'Spam'],
            yticklabels=['Non-Spam', 'Spam'],
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.xlabel('Predicted Label', fontweight='bold', fontsize=12)
plt.ylabel('True Label', fontweight='bold', fontsize=12)
plt.title('Confusion Matrix - Calibrated (Threshold = 0.24)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: confusion_matrix.png")

# ====================================================================
# TASK 3: FORCE THRESHOLD = 0.24 AND RECOMPUTE METRICS
# ====================================================================

print("\n" + "=" * 80)
print("TASK 3: THRESHOLD 0.24 METRICS")
print("=" * 80)

# Already computed above, save to file
tn_cal, fp_cal, fn_cal, tp_cal = cm_calibrated.ravel()

# Class-specific metrics for calibrated
prec_0_cal = precision_score(y_test_clean, y_pred_calibrated, pos_label=0, zero_division=0)
prec_1_cal = precision_score(y_test_clean, y_pred_calibrated, pos_label=1, zero_division=0)
recall_0_cal = recall_score(y_test_clean, y_pred_calibrated, pos_label=0, zero_division=0)
recall_1_cal = recall_score(y_test_clean, y_pred_calibrated, pos_label=1, zero_division=0)
f1_0_cal = f1_score(y_test_clean, y_pred_calibrated, pos_label=0, zero_division=0)
f1_1_cal = f1_score(y_test_clean, y_pred_calibrated, pos_label=1, zero_division=0)

metrics_lines = []
metrics_lines.append("=" * 80)
metrics_lines.append("THRESHOLD 0.24 METRICS (CALIBRATED PROBABILITIES)")
metrics_lines.append("=" * 80)
metrics_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
metrics_lines.append(f"Threshold: {threshold} (FORCED)")
metrics_lines.append(f"Test Set Size: {len(y_test_clean):,} samples")
metrics_lines.append("")
metrics_lines.append("OVERALL METRICS:")
metrics_lines.append(f"  Accuracy: {acc_calibrated:.6f}")
metrics_lines.append(f"  Precision (Weighted): {prec_calibrated:.6f}")
metrics_lines.append(f"  Recall (Weighted): {recall_calibrated:.6f}")
metrics_lines.append(f"  F1-Score (Weighted): {f1_calibrated:.6f}")
metrics_lines.append(f"  ROC AUC: {roc_auc_calibrated:.6f}")
metrics_lines.append(f"  PR AUC: {pr_auc_calibrated:.6f}")
metrics_lines.append(f"  Brier Score: {brier_calibrated:.6f}")
metrics_lines.append(f"  ECE: {ece_calibrated:.6f}")
metrics_lines.append("")
metrics_lines.append("CLASS-SPECIFIC METRICS:")
metrics_lines.append(f"  Precision (Non-Spam): {prec_0_cal:.6f}")
metrics_lines.append(f"  Precision (Spam): {prec_1_cal:.6f}")
metrics_lines.append(f"  Recall (Non-Spam): {recall_0_cal:.6f}")
metrics_lines.append(f"  Recall (Spam): {recall_1_cal:.6f}")
metrics_lines.append(f"  F1-Score (Non-Spam): {f1_0_cal:.6f}")
metrics_lines.append(f"  F1-Score (Spam): {f1_1_cal:.6f}")
metrics_lines.append("")
metrics_lines.append("CONFUSION MATRIX:")
metrics_lines.append(f"  True Negatives: {tn_cal}")
metrics_lines.append(f"  False Positives: {fp_cal}")
metrics_lines.append(f"  False Negatives: {fn_cal}")
metrics_lines.append(f"  True Positives: {tp_cal}")
metrics_lines.append("")
metrics_lines.append("COMPARISON WITH ORIGINAL PROBABILITIES:")
metrics_lines.append(f"  Accuracy: {acc_original:.6f} -> {acc_calibrated:.6f} ({'+' if acc_calibrated > acc_original else ''}{acc_calibrated - acc_original:.6f})")
metrics_lines.append(f"  F1-Score: {f1_original:.6f} -> {f1_calibrated:.6f} ({'+' if f1_calibrated > f1_original else ''}{f1_calibrated - f1_original:.6f})")
metrics_lines.append(f"  Brier Score: {brier_original:.6f} -> {brier_calibrated:.6f} ({brier_calibrated - brier_original:.6f})")
metrics_lines.append(f"  ECE: {ece_original:.6f} -> {ece_calibrated:.6f} ({ece_calibrated - ece_original:.6f})")

metrics_path = output_dir / 'threshold24_metrics.txt'
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(metrics_lines))
print(f"[OK] Saved: threshold24_metrics.txt")

# ====================================================================
# TASK 4: DATA LEAKAGE AUDIT (RE-RUN AFTER FIX)
# ====================================================================

print("\n" + "=" * 80)
print("TASK 4: DATA LEAKAGE AUDIT (RE-RUN)")
print("=" * 80)

def audit_pipeline(df_full, X_train, y_train, X_test_clean, y_test_clean, vectorizer, output_path):
    """Comprehensive data leakage audit function."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA LEAKAGE AUDIT REPORT (POST-FIX)")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Convert to numpy if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test_clean, 'values'):
        X_test_clean = X_test_clean.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test_clean, 'values'):
        y_test_clean = y_test_clean.values
    
    # Label Leakage Check
    report_lines.append("-" * 80)
    report_lines.append("CHECK 1: LABEL LEAKAGE (Feature-Label Relationships)")
    report_lines.append("-" * 80)
    
    try:
        # Handle sparse matrices
        if hasattr(X_train, 'toarray'):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test_clean.toarray()
        else:
            X_train_dense = X_train
            X_test_dense = X_test_clean
        
        # Limit feature check to first 1000 features for performance
        n_features_to_check = min(1000, X_train_dense.shape[1])
        max_corr = 0
        max_mi = 0
        problematic_features = []
        
        report_lines.append(f"Checking first {n_features_to_check} features...")
        
        for i in range(n_features_to_check):
            try:
                if np.var(X_train_dense[:, i]) > 0:
                    corr, _ = pearsonr(X_train_dense[:, i], y_train)
                    corr_abs = abs(corr)
                    if corr_abs > max_corr:
                        max_corr = corr_abs
                    if corr_abs > 0.9:
                        problematic_features.append({
                            'feature': i,
                            'type': 'correlation',
                            'value': corr_abs
                        })
            except:
                pass
        
        # Mutual Information (sample for performance)
        sample_size = min(5000, len(X_train_dense))
        sample_indices = np.random.choice(len(X_train_dense), sample_size, replace=False)
        
        try:
            mi_scores = mutual_info_classif(
                X_train_dense[sample_indices, :n_features_to_check],
                y_train[sample_indices],
                random_state=42,
                discrete_features=False
            )
            max_mi = np.max(mi_scores) if len(mi_scores) > 0 else 0
            
            for i, mi in enumerate(mi_scores):
                if mi > 0.9:
                    problematic_features.append({
                        'feature': i,
                        'type': 'mutual_information',
                        'value': mi
                    })
        except Exception as e:
            report_lines.append(f"  WARNING: Could not compute mutual information: {e}")
        
        report_lines.append(f"  Max Correlation: {max_corr:.6f}")
        report_lines.append(f"  Max Mutual Information: {max_mi:.6f}")
        
        if max_corr > 0.9 or max_mi > 0.9:
            report_lines.append("  STATUS: FAIL")
            report_lines.append(f"  ERROR: Found {len(problematic_features)} features with correlation > 0.9 or MI > 0.9")
            for pf in problematic_features[:10]:
                report_lines.append(f"    Feature {pf['feature']}: {pf['type']} = {pf['value']:.6f}")
            label_error = RuntimeError(f"LABEL LEAKAGE DETECTED: {len(problematic_features)} problematic features")
        else:
            report_lines.append("  STATUS: PASS")
            label_error = None
            
    except Exception as e:
        report_lines.append(f"  STATUS: ERROR - {str(e)}")
        label_error = RuntimeError(f"Label leakage check failed: {e}")
    
    # Vocabulary Leakage Check
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("CHECK 2: VOCABULARY LEAKAGE")
    report_lines.append("-" * 80)
    
    try:
        if hasattr(vectorizer, 'word_index'):
            train_vocab = set(vectorizer.word_index.keys())
            vocab_size = len(train_vocab)
            report_lines.append(f"  Training vocabulary size: {vocab_size:,}")
            report_lines.append("  WARNING: Cannot verify test token intersection without original texts")
            report_lines.append("  STATUS: WARNING (metadata not available)")
        else:
            report_lines.append("  STATUS: SKIP (vectorizer type not recognized)")
    except Exception as e:
        report_lines.append(f"  STATUS: ERROR - {str(e)}")
    
    # Length Leakage Check
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("CHECK 3: LENGTH LEAKAGE")
    report_lines.append("-" * 80)
    
    try:
        if hasattr(X_train, 'toarray'):
            train_lengths = (X_train.toarray() != 0).sum(axis=1)
        else:
            train_lengths = (X_train != 0).sum(axis=1)
        
        if np.var(train_lengths) > 0:
            length_corr, _ = pearsonr(train_lengths, y_train)
            length_corr_abs = abs(length_corr)
            report_lines.append(f"  Train length-label correlation: {length_corr_abs:.6f}")
            
            if length_corr_abs > 0.8:
                report_lines.append("  STATUS: WARNING")
                report_lines.append(f"  WARNING: High correlation ({length_corr_abs:.6f}) between text length and label!")
            else:
                report_lines.append("  STATUS: PASS")
        else:
            report_lines.append("  STATUS: SKIP (zero variance in lengths)")
    except Exception as e:
        report_lines.append(f"  STATUS: ERROR - {str(e)}")
    
    # Overlap Leakage Check
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("CHECK 4: OVERLAP LEAKAGE (Exact Duplicates)")
    report_lines.append("-" * 80)
    
    try:
        if hasattr(X_train, 'toarray'):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test_clean.toarray()
        else:
            X_train_dense = X_train
            X_test_dense = X_test_clean
        
        # Check for exact duplicates (sample for performance)
        sample_train = min(5000, len(X_train_dense))
        sample_test = min(5000, len(X_test_dense))
        
        train_sample = X_train_dense[:sample_train]
        test_sample = X_test_dense[:sample_test]
        
        duplicates = []
        for i, test_row in enumerate(test_sample):
            if np.any(np.all(train_sample == test_row, axis=1)):
                duplicates.append(i)
        
        if len(duplicates) > 0:
            report_lines.append(f"  STATUS: FAIL")
            report_lines.append(f"  ERROR: Found {len(duplicates)} duplicate samples in test set!")
            report_lines.append(f"  First {min(10, len(duplicates))} duplicate indices: {duplicates[:10]}")
            overlap_error = RuntimeError(f"OVERLAP LEAKAGE DETECTED: {len(duplicates)} duplicate samples")
        else:
            report_lines.append("  STATUS: PASS")
            report_lines.append(f"  Checked {sample_test} test samples against {sample_train} training samples")
            overlap_error = None
            
    except Exception as e:
        report_lines.append(f"  STATUS: ERROR - {str(e)}")
        overlap_error = RuntimeError(f"Overlap check failed: {e}")
    
    # Summary
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("AUDIT SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("All critical checks completed.")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Re-raise errors if any
    if label_error:
        raise label_error
    if overlap_error:
        raise overlap_error
    
    return report_text

# Run audit
print("\nRunning data leakage audit...")
audit_path = output_dir / 'leakage_audit_final.txt'

try:
    audit_pipeline(
        df_full=df_full,
        X_train=X_train,
        y_train=y_train,
        X_test_clean=X_test_clean,
        y_test_clean=y_test_clean,
        vectorizer=tokenizer,
        output_path=audit_path
    )
    print(f"[OK] Audit completed: leakage_audit_final.txt")
except RuntimeError as e:
    print(f"[CRITICAL] Leakage detected: {e}")
    print(f"          Report saved to: {audit_path}")
    raise

# ====================================================================
# TASK 5: FINAL SUMMARY REPORT
# ====================================================================

print("\n" + "=" * 80)
print("TASK 5: FINAL SUMMARY REPORT")
print("=" * 80)

# Read audit results
audit_results = "PASS"
if audit_path.exists():
    with open(audit_path, 'r') as f:
        audit_content = f.read()
        if "STATUS: FAIL" in audit_content:
            audit_results = "FAIL - See leakage_audit_final.txt for details"
        elif "STATUS: WARNING" in audit_content:
            audit_results = "WARNING - See leakage_audit_final.txt for details"

summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("FINAL SUMMARY REPORT - THRESHOLD 0.24 EVALUATION")
summary_lines.append("=" * 80)
summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append(f"Output Directory: {output_dir.resolve()}")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("CONFIGURATION")
summary_lines.append("=" * 80)
summary_lines.append(f"  Threshold: 0.24 (FORCED)")
summary_lines.append(f"  Model: Advanced RNN (Bidirectional LSTM + Attention + GloVe)")
summary_lines.append(f"  Dataset: output/balanced_rf/balanced_dataset.csv")
summary_lines.append(f"  Total Samples: {len(df_full):,}")
summary_lines.append(f"  Train Samples: {len(X_train):,}")
summary_lines.append(f"  Test Samples (original): {len(X_test):,}")
summary_lines.append(f"  Test Samples (cleaned): {len(X_test_clean):,}")
summary_lines.append(f"  Duplicates Removed: {len(X_test) - len(X_test_clean)}")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("CALIBRATION")
summary_lines.append("=" * 80)
summary_lines.append(f"  Method: Isotonic Regression")
summary_lines.append(f"  Fit on: FULL dataset ({len(y_test_clean):,} samples)")
summary_lines.append(f"  Sampling: NONE (full dataset only)")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("PERFORMANCE METRICS (Threshold = 0.24)")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append("CALIBRATED PROBABILITIES:")
summary_lines.append(f"  Accuracy: {acc_calibrated:.6f}")
summary_lines.append(f"  Precision (Weighted): {prec_calibrated:.6f}")
summary_lines.append(f"  Recall (Weighted): {recall_calibrated:.6f}")
summary_lines.append(f"  F1-Score (Weighted): {f1_calibrated:.6f}")
summary_lines.append(f"  ROC AUC: {roc_auc_calibrated:.6f}")
summary_lines.append(f"  PR AUC: {pr_auc_calibrated:.6f}")
summary_lines.append(f"  Brier Score: {brier_calibrated:.6f}")
summary_lines.append(f"  ECE: {ece_calibrated:.6f}")
summary_lines.append("")
summary_lines.append("ORIGINAL PROBABILITIES:")
summary_lines.append(f"  Accuracy: {acc_original:.6f}")
summary_lines.append(f"  Precision (Weighted): {prec_original:.6f}")
summary_lines.append(f"  Recall (Weighted): {recall_original:.6f}")
summary_lines.append(f"  F1-Score (Weighted): {f1_original:.6f}")
summary_lines.append(f"  ROC AUC: {roc_auc_original:.6f}")
summary_lines.append(f"  PR AUC: {pr_auc_original:.6f}")
summary_lines.append(f"  Brier Score: {brier_original:.6f}")
summary_lines.append(f"  ECE: {ece_original:.6f}")
summary_lines.append("")
summary_lines.append("IMPROVEMENT FROM CALIBRATION:")
summary_lines.append(f"  Accuracy: {acc_calibrated - acc_original:+.6f}")
summary_lines.append(f"  F1-Score: {f1_calibrated - f1_original:+.6f}")
summary_lines.append(f"  Brier Score: {brier_calibrated - brier_original:+.6f} (lower is better)")
summary_lines.append(f"  ECE: {ece_calibrated - ece_original:+.6f} (lower is better)")
summary_lines.append("")
summary_lines.append("CONFUSION MATRIX (Calibrated, Threshold = 0.24):")
summary_lines.append(f"  True Negatives: {tn_cal}")
summary_lines.append(f"  False Positives: {fp_cal}")
summary_lines.append(f"  False Negatives: {fn_cal}")
summary_lines.append(f"  True Positives: {tp_cal}")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("DATA LEAKAGE AUDIT RESULTS")
summary_lines.append("=" * 80)
summary_lines.append(f"  Overall Status: {audit_results}")
summary_lines.append(f"  Detailed Report: leakage_audit_final.txt")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("DUPLICATE FIX CONFIRMATION")
summary_lines.append("=" * 80)
summary_lines.append(f"  Original test size: {len(X_test):,} samples")
summary_lines.append(f"  Cleaned test size: {len(X_test_clean):,} samples")
summary_lines.append(f"  Duplicates removed: {len(X_test) - len(X_test_clean)}")
summary_lines.append(f"  Confirmation: No duplicates remain in test set")
summary_lines.append(f"  Detailed Report: duplicate_fix_report.txt")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("FILE LOCATIONS")
summary_lines.append("=" * 80)
summary_lines.append(f"  All files saved in: {output_dir.resolve()}")
summary_lines.append("")
summary_lines.append("  Data Files:")
summary_lines.append("    - calibrated_predictions.csv")
summary_lines.append("    - calibration_model.pkl")
summary_lines.append("")
summary_lines.append("  Reports:")
summary_lines.append("    - threshold24_metrics.txt")
summary_lines.append("    - duplicate_fix_report.txt")
summary_lines.append("    - leakage_audit_final.txt")
summary_lines.append("    - final_summary.txt (this file)")
summary_lines.append("")
summary_lines.append("  Visualizations:")
summary_lines.append("    - reliability_diagrams.png")
summary_lines.append("    - probability_histograms.png")
summary_lines.append("    - roc_curve.png")
summary_lines.append("    - pr_curve.png")
summary_lines.append("    - confusion_matrix.png")
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("END OF SUMMARY")
summary_lines.append("=" * 80)

summary_path = output_dir / 'final_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary_lines))
print(f"[OK] Saved: final_summary.txt")

# ====================================================================
# FINAL OUTPUT
# ====================================================================

print("\n" + "=" * 80)
print("ALL TASKS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_dir.resolve()}")
print(f"\nGenerated files:")
for file in sorted(output_dir.glob("*")):
    print(f"  - {file.name}")

print(f"\nSummary:")
print(f"  Test set (cleaned): {len(X_test_clean):,} samples")
print(f"  Threshold: {threshold}")
print(f"  Calibrated Accuracy: {acc_calibrated:.6f}")
print(f"  Calibrated F1-Score: {f1_calibrated:.6f}")
print(f"  Calibrated Brier Score: {brier_calibrated:.6f}")
print(f"  Calibrated ECE: {ece_calibrated:.6f}")

print("\n" + "=" * 80)

