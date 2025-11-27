"""
Quick test script for SHL Recommendation System
Tests the in-memory FAISS search directly
"""

import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from shl_agent.tools.search_tool import search_assessments

def test_search():
    """Test search functionality"""
    print("=" * 60)
    print("Testing SHL Assessment Search")
    print("=" * 60)
    
    test_queries = [
        "Data Science assessments",
        "Leadership and management",
        "Python programming",
        "Customer service roles",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        results = search_assessments(query, max_results=3)
        print(results)
        print()

if __name__ == "__main__":
    test_search()
