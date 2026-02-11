import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models import predict_model_1, predict_model_2

def benchmark_model(predict_fn, model_name, df):
    """
    Benchmark a sentiment analysis model.
    
    Args:
        predict_fn: Function that takes text and returns (label, score)
        model_name: Name of the model for reporting
        df: DataFrame with 'review' and 'sentiment' columns
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")
    
    predictions = []
    scores = []
    start_time = time.time()
    
    for idx, row in df.iterrows():
        try:
            label, score = predict_fn(row["review"])
            # Normalize to lowercase for consistent comparison
            predictions.append(label.lower())
            scores.append(score)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            predictions.append("negative")  # default fallback
            scores.append(0.5)
    
    total_time = time.time() - start_time
    avg_time_per_review = total_time / len(df)
    
    # Calculate metrics
    accuracy = accuracy_score(df["sentiment"], predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["sentiment"], 
        predictions, 
        average='binary',
        pos_label='positive'
    )
    
    cm = confusion_matrix(df["sentiment"], predictions)
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"Total reviews: {len(df)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per review: {avg_time_per_review:.4f}s")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        "review": df["review"],
        "true_sentiment": df["sentiment"],
        "predicted_sentiment": predictions,
        "confidence_score": scores
    })
    
    output_file = f"./output/{model_name.replace(' ', '_').replace('/', '-')}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_time": total_time,
        "avg_time_per_review": avg_time_per_review,
        "confusion_matrix": cm.tolist()
    }


def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('./data/IMDB-movie-reviews.csv', encoding='latin-1', delimiter=';')
    print(f"Loaded {len(df)} reviews")
    
    # Benchmark both models
    results = []
    
    # Model 1: DistilBERT
    results.append(benchmark_model(
        predict_model_1, 
        "Model 1 - DistilBERT",
        df
    ))
    
    # Model 2: RoBERTa Large
    results.append(benchmark_model(
        predict_model_2,
        "Model 2 - RoBERTa Large",
        df
    ))
    
    # Create comparison summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'avg_time_per_review']]
    
    print("\n", comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('./output/benchmark_comparison.csv', index=False)
    print("\n\nBenchmark comparison saved to: ./output/benchmark_comparison.csv")
    
    # Determine best model
    best_accuracy = comparison_df.loc[comparison_df['accuracy'].idxmax()]
    best_f1 = comparison_df.loc[comparison_df['f1_score'].idxmax()]
    fastest = comparison_df.loc[comparison_df['avg_time_per_review'].idxmin()]
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"Best Accuracy:  {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best F1-Score:  {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
    print(f"Fastest:        {fastest['model_name']} ({fastest['avg_time_per_review']:.4f}s/review)")


if __name__ == "__main__":
    main()
