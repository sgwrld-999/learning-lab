"""
Search tool for SHL assessments using FAISS in-memory vector search
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class InMemoryFAISSSearcher:
    """In-memory FAISS searcher for assessment recommendations"""
    
    def __init__(self, data_path: str = None):
        """Initialize searcher with in-memory FAISS index"""
        if data_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "data" / "individual-assessment.json"
        
        self.data_path = Path(data_path)
        
        # Load data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        
        # Initialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build in-memory FAISS index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index in memory"""
        # Prepare text for embeddings
        texts = []
        for assessment in self.assessments:
            text = f"{assessment['title']} {assessment['description']}"
            if 'job_levels' in assessment and assessment['job_levels']:
                text += f" {' '.join(assessment['job_levels'])}"
            texts.append(text)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} assessments...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype='float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index (IndexIDMap for ID mapping)
        dimension = embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.index = faiss.IndexIDMap(base_index)
        
        # Add vectors with IDs
        ids = np.array(range(len(self.assessments)), dtype='int64')
        self.index.add_with_ids(embeddings, ids)
        
        print(f"✓ FAISS index built: {len(self.assessments)} assessments indexed")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for assessments using semantic search
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of assessment dictionaries
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid result
                assessment = self.assessments[int(idx)].copy()
                assessment['similarity_score'] = float(distance)
                results.append(assessment)
        
        return results


# Global searcher instance (lazy initialization)
_searcher = None


def get_searcher() -> InMemoryFAISSSearcher:
    """Get or create global searcher instance"""
    global _searcher
    if _searcher is None:
        _searcher = InMemoryFAISSSearcher()
    return _searcher


def search_assessments(query: str, max_results: int = 10) -> str:
    """
    Search for SHL assessments based on query
    
    This tool searches through 348+ SHL assessments using semantic search
    to find the most relevant assessments for the given query.
    
    Args:
        query: Search query describing requirements (e.g., "Data Science assessments")
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        JSON string containing list of relevant assessments with details
    """
    try:
        searcher = get_searcher()
        results = searcher.search(query, k=max_results)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "duration": result.get("duration"),
                "test_type": result.get("test_type", []),
                "job_levels": result.get("job_levels", []),
                "remote_testing": result.get("remote_testing", ""),
                "adaptive": result.get("adaptive", ""),
                "similarity_score": result.get("similarity_score", 0.0)
            })
        
        return json.dumps(formatted_results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})
