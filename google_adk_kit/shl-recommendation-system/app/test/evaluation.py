"""
Evaluation Script for SHL Assessment Recommendation System
Computes Mean Recall@K metrics on test dataset
"""

import json
import pandas as pd
from typing import List, Dict, Set
from shl_agent.tools.search_tool import search_assessments


def load_test_data(file_path: str) -> List[Dict]:
    """Load test dataset with queries and ground truth assessments"""
    # This should load your labeled test set
    # Format: [{query: str, relevant_urls: List[str]}, ...]
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    
    test_data = []
    for _, row in df.iterrows():
        test_data.append({
            'query': row['query'],
            'relevant_urls': row['relevant_urls'].split(',') if isinstance(row['relevant_urls'], str) else []
        })
    
    return test_data


def recall_at_k(predicted_urls: List[str], relevant_urls: List[str], k: int) -> float:
    """
    Calculate Recall@K
    
    Recall@K = (Number of relevant items in top K) / (Total relevant items)
    """
    if not relevant_urls:
        return 0.0
    
    predicted_top_k = set(predicted_urls[:k])
    relevant_set = set(relevant_urls)
    
    relevant_in_top_k = len(predicted_top_k.intersection(relevant_set))
    total_relevant = len(relevant_set)
    
    return relevant_in_top_k / total_relevant


def mean_recall_at_k(test_data: List[Dict], k: int = 10) -> float:
    """
    Calculate Mean Recall@K across all test queries
    
    MeanRecall@K = (1/N) * Σ Recall@K_i
    """
    recalls = []
    
    for i, test_item in enumerate(test_data):
        query = test_item['query']
        relevant_urls = test_item['relevant_urls']
        
        print(f"\nQuery {i+1}/{len(test_data)}: {query}")
        
        # Get predictions from system
        results_json = search_assessments(query, max_results=k)
        results = json.loads(results_json)
        
        predicted_urls = [r['url'] for r in results if 'url' in r]
        
        # Calculate recall for this query
        recall = recall_at_k(predicted_urls, relevant_urls, k)
        recalls.append(recall)
        
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  Predicted: {len(predicted_urls)} | Relevant: {len(relevant_urls)} | Matched: {len(set(predicted_urls[:k]).intersection(set(relevant_urls)))}")
    
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return mean_recall


def evaluate_system(test_file_path: str, output_file: str = 'evaluation_results.json'):
    """
    Run full evaluation on test dataset
    """
    print("=" * 80)
    print("SHL Assessment Recommendation System - Evaluation")
    print("=" * 80)
    
    # Load test data
    print(f"\nLoading test data from: {test_file_path}")
    test_data = load_test_data(test_file_path)
    print(f"Loaded {len(test_data)} test queries")
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("Computing Metrics...")
    print("=" * 80)
    
    recall_at_5 = mean_recall_at_k(test_data, k=5)
    recall_at_10 = mean_recall_at_k(test_data, k=10)
    
    # Results
    results = {
        'mean_recall_at_5': recall_at_5,
        'mean_recall_at_10': recall_at_10,
        'num_queries': len(test_data)
    }
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Mean Recall@5:  {recall_at_5:.4f} ({recall_at_5*100:.2f}%)")
    print(f"Mean Recall@10: {recall_at_10:.4f} ({recall_at_10*100:.2f}%)")
    print(f"Number of test queries: {len(test_data)}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default test file path
    test_file = "../data/test_labeled.xlsx"  # Update with your actual test file
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    try:
        results = evaluate_system(test_file)
    except FileNotFoundError:
        print(f"\nError: Test file not found: {test_file}")
        print("\nUsage: python evaluation.py [test_file_path]")
        print("\nPlease provide the labeled test dataset with columns: 'query' and 'relevant_urls'")
