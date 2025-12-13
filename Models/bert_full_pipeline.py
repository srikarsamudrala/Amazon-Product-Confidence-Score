"""
Full BERT Sentiment Postprocessing Pipeline
Implements complete workflow with validation and preview before final output.
"""

import pandas as pd
import sys
from pathlib import Path


def load_and_validate(df):
    """SECTION 1: Load and validate the dataset."""
    print("=" * 60)
    print("SECTION 1: LOAD AND VALIDATE")
    print("=" * 60)
    
    # Validate required columns exist
    required_cols = ['asin', 'review_text', 'overall', 'is_spam', 
                     'bert_p1', 'bert_p2', 'bert_p3', 'bert_p4', 'bert_p5']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ All required columns present")
    print(f"  Loaded {len(df)} rows")
    
    # Validate bert_p1...bert_p5 are numeric and non-NaN
    prob_cols = ['bert_p1', 'bert_p2', 'bert_p3', 'bert_p4', 'bert_p5']
    print("\nValidating probability columns...")
    for col in prob_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError as e:
            raise ValueError(f"Column {col} contains non-numeric values: {e}")
        
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(f"Column {col} contains {nan_count} NaN values")
    
    print(f"✓ All probability columns are numeric and non-NaN")
    
    # Validate probability sums
    print("\nValidating probability sums...")
    prob_sum = df[prob_cols].sum(axis=1)
    
    invalid_mask = (prob_sum < 0.999) | (prob_sum > 1.001)
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        invalid_sums = prob_sum[invalid_mask].tolist()[:10]
        raise ValueError(
            f"Found {invalid_count} rows where probabilities do not sum to 0.999-1.001. "
            f"Example sums: {invalid_sums}"
        )
    
    print(f"✓ All {len(df)} rows have valid probability sums (0.999-1.001)")
    
    return df


def spam_filtering(df, all_asins):
    """SECTION 2: Filter out spam reviews but keep all ASINs."""
    print("\n" + "=" * 60)
    print("SECTION 2: SPAM FILTERING")
    print("=" * 60)
    
    initial_count = len(df)
    initial_asins = df['asin'].nunique()
    
    # Keep only non-spam rows (is_spam == 0) for processing
    df_filtered = df[df['is_spam'] == 0].copy()
    
    filtered_count = len(df_filtered)
    filtered_asins = df_filtered['asin'].nunique()
    
    print(f"Initial rows: {initial_count}")
    print(f"After spam filtering: {filtered_count} rows (removed {initial_count - filtered_count} spam reviews)")
    print(f"Initial ASINs: {initial_asins}")
    print(f"ASINs with non-spam reviews: {filtered_asins}")
    
    # Note: ASINs with zero non-spam reviews will be handled in aggregation
    if filtered_count == 0:
        raise ValueError("No non-spam reviews remaining after filtering")
    
    if filtered_asins == 0:
        raise ValueError("No ASINs remaining after spam removal")
    
    print(f"✓ Spam filtering complete - {filtered_asins} ASINs have non-spam reviews")
    if initial_asins > filtered_asins:
        print(f"  Note: {initial_asins - filtered_asins} ASINs have no non-spam reviews (will be included with NaN values)")
    
    return df_filtered


def per_review_computations(df):
    """SECTION 3: Compute per-review metrics."""
    print("\n" + "=" * 60)
    print("SECTION 3: PER-REVIEW COMPUTATIONS")
    print("=" * 60)
    
    # Compute expected_sentiment_rating: E = 1*p1 + 2*p2 + 3*p3 + 4*p4 + 5*p5
    print("Computing expected_sentiment_rating...")
    df['expected_sentiment_rating'] = (
        1 * df['bert_p1'] +
        2 * df['bert_p2'] +
        3 * df['bert_p3'] +
        4 * df['bert_p4'] +
        5 * df['bert_p5']
    )
    
    # Compute confidence_score_20_100: 20 * E
    print("Computing confidence_score_20_100...")
    df['confidence_score_20_100'] = 20 * df['expected_sentiment_rating']
    
    print(f"✓ Computed metrics for {len(df)} reviews")
    print(f"  expected_sentiment_rating range: {df['expected_sentiment_rating'].min():.4f} - {df['expected_sentiment_rating'].max():.4f}")
    print(f"  confidence_score_20_100 range: {df['confidence_score_20_100'].min():.2f} - {df['confidence_score_20_100'].max():.2f}")
    
    return df


def asin_aggregation(df, all_asins):
    """SECTION 4: Aggregate per ASIN, including all ASINs from input."""
    print("\n" + "=" * 60)
    print("SECTION 4: ASIN-LEVEL AGGREGATION")
    print("=" * 60)
    
    print("Aggregating metrics per ASIN...")
    
    agg_df = df.groupby('asin').agg({
        'overall': 'mean',  # asin_real_rating
        'expected_sentiment_rating': 'mean',  # asin_expected_sentiment_rating
        'confidence_score_20_100': 'mean',  # asin_confidence_score_20_100
        'asin': 'count'  # review_count
    }).rename(columns={
        'overall': 'asin_real_rating',
        'expected_sentiment_rating': 'asin_expected_sentiment_rating',
        'confidence_score_20_100': 'asin_confidence_score_20_100',
        'asin': 'review_count'
    }).reset_index()
    
    # Add ASINs with zero non-spam reviews (they will have NaN values)
    asins_with_data = set(agg_df['asin'].unique())
    asins_missing = set(all_asins) - asins_with_data
    
    if asins_missing:
        print(f"Adding {len(asins_missing)} ASINs with no non-spam reviews (will have NaN values)")
        missing_df = pd.DataFrame({
            'asin': list(asins_missing),
            'asin_real_rating': [None] * len(asins_missing),
            'asin_expected_sentiment_rating': [None] * len(asins_missing),
            'asin_confidence_score_20_100': [None] * len(asins_missing),
            'review_count': [0] * len(asins_missing)
        })
        agg_df = pd.concat([agg_df, missing_df], ignore_index=True)
    
    print(f"✓ Aggregated data for {len(agg_df)} ASINs (all ASINs from input)")
    
    return agg_df, df


def compute_product_confidence(agg_df):
    """SECTION 5: Compute final product confidence score."""
    print("\n" + "=" * 60)
    print("SECTION 5: FINAL PRODUCT SCORE")
    print("=" * 60)
    
    # First compute weighted rating on 1-5 scale: R = 0.65 * asin_real_rating + 0.35 * asin_expected_sentiment_rating
    print("Computing weighted rating (1-5 scale)...")
    weighted_rating = (
        0.65 * agg_df['asin_real_rating'] + 
        0.35 * agg_df['asin_expected_sentiment_rating']
    )
    
    # Then scale to 20-100: product_confidence_score = 20 * R
    print("Scaling to 20-100 range...")
    agg_df['product_confidence_score'] = 20 * weighted_rating
    
    valid_asins = agg_df['product_confidence_score'].notna().sum()
    print(f"✓ Computed product_confidence_score for {valid_asins} ASINs with valid data")
    if valid_asins < len(agg_df):
        print(f"  {len(agg_df) - valid_asins} ASINs have NaN (no non-spam reviews)")
    valid_scores = agg_df['product_confidence_score'].dropna()
    if len(valid_scores) > 0:
        print(f"  Range: {valid_scores.min():.2f} - {valid_scores.max():.2f} (20-100 scale)")
    
    return agg_df


def show_preview(agg_df):
    """SECTION 7: Show preview of ALL ASINs."""
    print("\n" + "=" * 60)
    print("SECTION 7: PREVIEW - ALL ASINs")
    print("=" * 60)
    
    # Sort by asin alphabetically - show ALL ASINs
    preview_df = agg_df.sort_values('asin').copy()
    
    # Select required columns for preview
    preview = preview_df[['asin', 'asin_real_rating', 'asin_expected_sentiment_rating', 
                          'product_confidence_score']].copy()
    
    print(f"\nPreview Table (ALL {len(preview_df)} ASINs, sorted alphabetically):")
    print("=" * 80)
    print(preview.to_string(index=False))
    print("=" * 80)
    
    return preview_df


def generate_outputs(agg_df, review_df):
    """SECTION 6: Generate final CSV outputs."""
    print("\n" + "=" * 60)
    print("SECTION 6: GENERATING FINAL CSV OUTPUTS")
    print("=" * 60)
    
    # A. Per-review CSV
    print("\nCreating per-review CSV...")
    per_review_cols = ['asin', 'overall', 'expected_sentiment_rating', 'confidence_score_20_100']
    per_review_output = review_df[per_review_cols].copy()
    per_review_output.to_csv('bert_pipeline_per_review.csv', index=False)
    print(f"✓ Saved: bert_pipeline_per_review.csv")
    print(f"  Rows: {len(per_review_output)}, Columns: {len(per_review_output.columns)}")
    
    # B. Per-ASIN CSV
    print("\nCreating per-ASIN CSV...")
    asin_cols = ['asin', 'asin_real_rating', 'asin_expected_sentiment_rating', 
                 'asin_confidence_score_20_100', 'product_confidence_score', 'review_count']
    asin_output = agg_df[asin_cols].copy()
    asin_output.to_csv('bert_pipeline_per_asin.csv', index=False)
    print(f"✓ Saved: bert_pipeline_per_asin.csv")
    print(f"  ASINs: {len(asin_output)}, Columns: {len(asin_output.columns)}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def main():
    """Main workflow."""
    print("BERT SENTIMENT POSTPROCESSING PIPELINE")
    print("=" * 60)
    
    # Get input file
    if len(sys.argv) < 2:
        input_file = input("Enter path to input CSV file: ").strip()
    else:
        input_file = sys.argv[1]
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"\nReading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Get all unique ASINs before filtering
    all_asins = sorted(df['asin'].unique())
    print(f"Total unique ASINs in input: {len(all_asins)}")
    
    # SECTION 1: Load and validate
    df = load_and_validate(df)
    
    # SECTION 2: Spam filtering
    df = spam_filtering(df, all_asins)
    
    # SECTION 3: Per-review computations
    df = per_review_computations(df)
    
    # SECTION 4: ASIN aggregation (include all ASINs)
    agg_df, review_df = asin_aggregation(df, all_asins)
    
    # SECTION 5: Product confidence score
    agg_df = compute_product_confidence(agg_df)
    
    # SECTION 7: Show preview
    show_preview(agg_df)
    
    # SECTION 6: Generate outputs (automatically for all ASINs)
    generate_outputs(agg_df, review_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

