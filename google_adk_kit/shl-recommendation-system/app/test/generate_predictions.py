"""
Generate predictions CSV file for SHL Assessment Recommendation System
Format: Query, Assessment_url (as per submission requirements)
"""

import json
import pandas as pd
from typing import List, Dict
from shl_agent.tools.search_tool import search_assessments


def load_unlabeled_test_queries(file_path: str) -> List[str]:
    """Load unlabeled test queries"""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # Assume text file with one query per line
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    # Assuming column name is 'query' or first column
    if 'query' in df.columns:
        return df['query'].tolist()
    else:
        return df.iloc[:, 0].tolist()


def generate_predictions(queries: List[str], max_results: int = 10) -> List[Dict[str, str]]:
    """
    Generate predictions for all queries
    
    Returns:
        List of dicts with 'Query' and 'Assessment_url' keys
    """
    predictions = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}: {query[:60]}...")
        
        try:
            # Get recommendations
            results_json = search_assessments(query, max_results=max_results)
            results = json.loads(results_json)
            
            # Extract URLs
            for result in results:
                if 'url' in result and result['url']:
                    predictions.append({
                        'Query': query,
                        'Assessment_url': result['url']
                    })
            
            print(f"  ✓ Found {len(results)} recommendations")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return predictions


def generate_predictions_csv(
    test_queries_file: str,
    output_csv: str = 'predictions.csv',
    max_results: int = 10
):
    """
    Generate predictions CSV file for submission
    
    Args:
        test_queries_file: Path to file containing unlabeled test queries
        output_csv: Output CSV file path
        max_results: Maximum recommendations per query (default: 10)
    """
    print("=" * 80)
    print("SHL Assessment Recommendation - Predictions Generator")
    print("=" * 80)
    
    # Load queries
    print(f"\nLoading test queries from: {test_queries_file}")
    queries = load_unlabeled_test_queries(test_queries_file)
    print(f"Loaded {len(queries)} queries")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("Generating Predictions...")
    print("=" * 80)
    
    predictions = generate_predictions(queries, max_results=max_results)
    
    # Create DataFrame and save
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")
    print(f"Queries processed: {len(queries)}")
    print(f"Average recommendations per query: {len(predictions) / len(queries) if queries else 0:.1f}")
    print(f"\nPredictions saved to: {output_csv}")
    print("\nFormat:")
    print("  Query, Assessment_url")
    print("  Query 1, Recommendation 1 URL")
    print("  Query 1, Recommendation 2 URL")
    print("  ...")
    
    # Show preview
    print("\nPreview (first 5 rows):")
    print(df.head().to_string(index=False))
    
    return df


if __name__ == "__main__":
    import sys
    import os
    
    # Default paths
    test_file = "../data/test_unlabeled.xlsx"  # Update with your test file
    output_file = "predictions.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        if not os.path.exists(test_file):
            print(f"\nError: Test file not found: {test_file}")
            print("\nUsage: python generate_predictions.py [test_file] [output_csv]")
            print("\nTest file should contain queries (one per row)")
            print("Supported formats: .xlsx, .csv, .txt")
            sys.exit(1)
        
        df = generate_predictions_csv(test_file, output_file, max_results=10)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
