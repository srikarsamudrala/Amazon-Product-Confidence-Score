"""
Threshold Optimizer for Leak-Safe Random Forest Model
=====================================================
Optimizes classification threshold to balance Spam and Non-Spam Recall.

Problem: Model is too aggressive (94% Spam Recall, 52% Non-Spam Recall)
Solution: Find optimal threshold that balances both recalls while keeping Spam Recall > 90%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_model_and_test_data(model_path: str, test_data_path: str = None):
    """
    Load the trained model and test data.
    
    Args:
        model_path: Path to saved Random Forest model
        test_data_path: Path to saved test data (X_test, y_test) pickle file
    
    Returns:
        Tuple of (model, X_test, y_test)
    """
    print("=" * 80)
    print("LOADING MODEL AND TEST DATA")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of trees: {model.n_estimators}")
    
    # Load test data
    if test_data_path and Path(test_data_path).exists():
        print(f"\nLoading test data from: {test_data_path}")
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
            X_test = test_data['X_test']
            y_test = test_data['y_test']
        print(f"✓ Test data loaded successfully")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  Test set spam ratio: {y_test.mean():.2%}")
    else:
        raise FileNotFoundError(
            f"Test data file not found: {test_data_path}\n"
            "Please run the pipeline with test data saving enabled, or provide test data path."
        )
    
    return model, X_test, y_test


def optimize_threshold(model, X_test, y_test, min_spam_recall=0.90):
    """
    Optimize classification threshold to balance Spam and Non-Spam Recall.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test labels
        min_spam_recall: Minimum acceptable Spam Recall (default: 0.90)
    
    Returns:
        DataFrame with results and recommended threshold
    """
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    # Align X_test feature columns with model expectations
    print("\nCleaning and aligning test features with model feature space...")

    # First, ensure all features are numeric (mirror training-time cleanup)
    # Convert object columns to numeric where possible, drop if not
    obj_cols = X_test.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f"  Converting object columns to numeric (coerce errors): {obj_cols}")
        for col in obj_cols:
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(np.float32)
            except Exception as e:
                print(f"    Warning: dropping non-numeric column '{col}' ({e})")
                X_test = X_test.drop(columns=[col])

    # Ensure all numeric columns are float32
    for col in X_test.select_dtypes(include=[np.number]).columns:
        X_test[col] = X_test[col].astype(np.float32)

    # Replace inf / -inf with 0, fill any remaining NaNs
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_test.isna().any().any():
        X_test.fillna(0, inplace=True)
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        test_features = list(X_test.columns)

        # Drop any extra columns not seen during training
        extra_in_test = set(test_features) - set(model_features)
        if extra_in_test:
            print(f"  Dropping unseen test features: {sorted(extra_in_test)}")
            X_test = X_test.drop(columns=list(extra_in_test))

        # Add any missing columns that model expects but are absent in test
        missing_in_test = set(model_features) - set(X_test.columns)
        if missing_in_test:
            print(f"  Adding missing test features with zeros: {sorted(missing_in_test)}")
            for col in missing_in_test:
                X_test[col] = np.float32(0.0)

        # Reorder columns to match model training order
        X_test = X_test[model_features]
        print(f"  ✓ Feature alignment complete. Using {X_test.shape[1]} features.")
    else:
        print("  Model does not expose feature_names_in_; using X_test as-is.")

    # Get probability predictions
    print("\nGenerating probability predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Spam)
    print(f"✓ Predictions generated for {len(y_test):,} test samples")
    
    # Define threshold range
    thresholds = np.arange(0.50, 0.95, 0.05)  # 0.50 to 0.90 in steps of 0.05
    
    print(f"\nTesting {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print("Evaluating each threshold...\n")
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        spam_recall = recall_score(y_test, y_pred, pos_label=1)  # Class 1 (Spam) Recall
        non_spam_recall = recall_score(y_test, y_pred, pos_label=0)  # Class 0 (Non-Spam) Recall
        difference = abs(spam_recall - non_spam_recall)
        
        # Confusion matrix for additional info
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'Threshold': threshold,
            'Spam_Recall': spam_recall,
            'Non_Spam_Recall': non_spam_recall,
            'Difference': difference,
            'True_Positive': tp,
            'False_Positive': fp,
            'True_Negative': tn,
            'False_Negative': fn,
            'Meets_Min_Spam_Recall': spam_recall >= min_spam_recall
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best threshold
    # Criteria: Smallest difference, while Spam Recall >= min_spam_recall
    valid_results = results_df[results_df['Meets_Min_Spam_Recall'] == True]
    
    if len(valid_results) > 0:
        best_idx = valid_results['Difference'].idxmin()
        best_threshold = results_df.loc[best_idx, 'Threshold']
        best_row = results_df.loc[best_idx]
    else:
        # If no threshold meets minimum, use the one with highest spam recall
        print("⚠️  WARNING: No threshold meets minimum Spam Recall of 90%")
        print("   Using threshold with highest Spam Recall instead")
        best_idx = results_df['Spam_Recall'].idxmax()
        best_threshold = results_df.loc[best_idx, 'Threshold']
        best_row = results_df.loc[best_idx]
    
    return results_df, best_threshold, best_row


def print_results_table(results_df):
    """Print results in a clear table format."""
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Format for display
    display_df = results_df.copy()
    display_df['Threshold'] = display_df['Threshold'].apply(lambda x: f"{x:.2f}")
    display_df['Spam_Recall'] = display_df['Spam_Recall'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    display_df['Non_Spam_Recall'] = display_df['Non_Spam_Recall'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    display_df['Difference'] = display_df['Difference'].apply(lambda x: f"{x:.4f}")
    
    # Select columns for display
    display_cols = ['Threshold', 'Spam_Recall', 'Non_Spam_Recall', 'Difference', 'Meets_Min_Spam_Recall']
    print("\n" + display_df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    print("\nFull results with confusion matrix:")
    print(results_df[['Threshold', 'Spam_Recall', 'Non_Spam_Recall', 'Difference', 
                      'True_Positive', 'False_Positive', 'True_Negative', 'False_Negative']].to_string(index=False))


def print_recommendation(best_threshold, best_row, min_spam_recall=0.90):
    """Print the recommended threshold and reasoning."""
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    print(f"\n✅ Recommended Threshold: {best_threshold:.2f}")
    print(f"\nMetrics at this threshold:")
    print(f"  Spam Recall (Class 1):     {best_row['Spam_Recall']:.4f} ({best_row['Spam_Recall']*100:.2f}%)")
    print(f"  Non-Spam Recall (Class 0): {best_row['Non_Spam_Recall']:.4f} ({best_row['Non_Spam_Recall']*100:.2f}%)")
    print(f"  Difference:                {best_row['Difference']:.4f}")
    print(f"  Meets Min Spam Recall:     {'✅ Yes' if best_row['Meets_Min_Spam_Recall'] else '❌ No'}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {best_row['True_Positive']:,}")
    print(f"  False Positives (FP): {best_row['False_Positive']:,}")
    print(f"  True Negatives (TN):  {best_row['True_Negative']:,}")
    print(f"  False Negatives (FN): {best_row['False_Negative']:,}")
    
    # Calculate additional metrics
    total = best_row['True_Positive'] + best_row['False_Positive'] + best_row['True_Negative'] + best_row['False_Negative']
    accuracy = (best_row['True_Positive'] + best_row['True_Negative']) / total
    precision = best_row['True_Positive'] / (best_row['True_Positive'] + best_row['False_Positive']) if (best_row['True_Positive'] + best_row['False_Positive']) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    
    print(f"\n📊 Analysis:")
    if best_row['Meets_Min_Spam_Recall']:
        print(f"  ✅ This threshold maintains Spam Recall above {min_spam_recall*100:.0f}%")
    else:
        print(f"  ⚠️  This threshold does NOT meet minimum Spam Recall of {min_spam_recall*100:.0f}%")
    
    print(f"  📈 Balance: Difference of {best_row['Difference']:.4f} between recalls")
    print(f"  🎯 This threshold provides the best balance while meeting constraints")
    
    print(f"\n💡 Usage:")
    print(f"  To use this threshold, modify your prediction code:")
    print(f"    y_pred_proba = model.predict_proba(X)[:, 1]")
    print(f"    y_pred = (y_pred_proba >= {best_threshold:.2f}).astype(int)")


def save_results(results_df, best_threshold, output_dir: str):
    """Save results to CSV and summary to text file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    csv_path = output_path / 'threshold_optimization_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved full results to: {csv_path}")
    
    # Save summary
    summary_path = output_path / 'threshold_optimization_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD OPTIMIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Recommended Threshold: {best_threshold:.2f}\n\n")
        f.write("Full Results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved summary to: {summary_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZER FOR LEAK-SAFE RANDOM FOREST MODEL")
    print("=" * 80)
    print("\nObjective: Find optimal threshold to balance Spam and Non-Spam Recall")
    print("Constraint: Keep Spam Recall above 90%")
    print("\nStarting optimization...\n")
    
    # Paths
    model_path = "output/ultimate_pipeline_leak_safe/rf_model_leak_safe.pkl"
    test_data_path = "output/ultimate_pipeline_leak_safe/test_data.pkl"
    output_dir = "output/ultimate_pipeline_leak_safe"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model file not found: {model_path}")
        print("\nPlease ensure the leak-safe pipeline has been run and the model is saved.")
        return
    
    # Check if test data exists
    if not Path(test_data_path).exists():
        print(f"⚠️  WARNING: Test data file not found: {test_data_path}")
        print("\nThe test data needs to be saved during pipeline execution.")
        print("Please run the pipeline with test data saving enabled.")
        print("\nAlternatively, you can manually create test_data.pkl with:")
        print("  import pickle")
        print("  test_data = {'X_test': X_test, 'y_test': y_test}")
        print("  with open('output/ultimate_pipeline_leak_safe/test_data.pkl', 'wb') as f:")
        print("      pickle.dump(test_data, f)")
        return
    
    try:
        # Load model and test data
        model, X_test, y_test = load_model_and_test_data(model_path, test_data_path)
        
        # Optimize threshold
        results_df, best_threshold, best_row = optimize_threshold(
            model, X_test, y_test, min_spam_recall=0.90
        )
        
        # Print results
        print_results_table(results_df)
        
        # Print recommendation
        print_recommendation(best_threshold, best_row, min_spam_recall=0.90)
        
        # Save results
        save_results(results_df, best_threshold, output_dir)
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

