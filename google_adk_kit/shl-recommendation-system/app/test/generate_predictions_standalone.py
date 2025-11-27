"""
Generate predictions CSV file for SHL Assessment Recommendation System
Standalone version that directly uses the search functionality
"""

import json
import pandas as pd
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import search functionality directly
from shl_agent.tools.search_tool import InMemoryFAISSSearcher


def load_queries_from_dataset(file_path: str) -> List[str]:
    """Load queries from dataset"""
    df = pd.read_excel(file_path)
    
    # Get unique queries
    queries = df['Query'].unique().tolist()
    return queries


def generate_predictions(queries: List[str], max_results: int = 10) -> List[Dict[str, str]]:
    """
    Generate predictions for all queries
    
    Returns:
        List of dicts with 'Query' and 'Assessment_url' keys
    """
    # Initialize searcher
    print("Initializing FAISS searcher...")
    data_path = os.path.join(os.path.dirname(__file__), '../data/individual-assessment.json')
    searcher = InMemoryFAISSSearcher(data_path)
    
    predictions = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}: {query[:80]}...")
        
        try:
            # Get recommendations
            results = searcher.search(query, k=max_results)
            
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


def main():
    """Main function"""
    print("=" * 80)
    print("SHL Assessment Recommendation - Predictions Generator")
    print("=" * 80)
    
    # Default paths
    test_file = "../data/Gen_AI_Dataset.xlsx"
    output_file = "predictions.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Load queries
    print(f"\nLoading test queries from: {test_file}")
    queries = load_queries_from_dataset(test_file)
    print(f"Loaded {len(queries)} unique queries")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("Generating Predictions...")
    print("=" * 80)
    
    predictions = generate_predictions(queries, max_results=10)
    
    # Create DataFrame and save
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")
    print(f"Unique queries processed: {len(queries)}")
    print(f"Average recommendations per query: {len(predictions) / len(queries) if queries else 0:.1f}")
    print(f"\nPredictions saved to: {output_file}")
    print("\nFormat:")
    print("  Query, Assessment_url")
    
    # Show preview
    print("\nPreview (first 10 rows):")
    print(df.head(10).to_string(index=False))
    
    return df


if __name__ == "__main__":
    try:
        df = main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
