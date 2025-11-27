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


def balance_test_types(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Balance test types in results for multi-domain queries.
    
    If query mentions both technical/skills AND behavioral/personality aspects,
    ensure balanced mix of K (Knowledge & Skills) and P (Personality) test types.
    """
    # Keywords indicating different test type needs
    technical_keywords = ['developer', 'programming', 'technical', 'coding', 'software', 
                         'python', 'java', 'sql', 'data', 'analyst', 'engineer']
    behavioral_keywords = ['collaborate', 'leadership', 'team', 'communication', 
                          'personality', 'behavior', 'soft skills', 'interpersonal']
    
    query_lower = query.lower()
    needs_technical = any(kw in query_lower for kw in technical_keywords)
    needs_behavioral = any(kw in query_lower for kw in behavioral_keywords)
    
    # If query needs both, ensure balanced results
    if needs_technical and needs_behavioral:
        k_type = []  # Knowledge & Skills
        p_type = []  # Personality & Behavior
        other_type = []
        
        for result in results:
            test_type = result.get('test_type', '')
            if isinstance(test_type, list):
                test_type = ','.join(test_type) if test_type else ''
            
            if 'K' in test_type:
                k_type.append(result)
            elif 'P' in test_type:
                p_type.append(result)
            else:
                other_type.append(result)
        
        # Balance: try to get 50-50 split or close to it
        target_each = len(results) // 2
        balanced = []
        balanced.extend(k_type[:target_each])
        balanced.extend(p_type[:target_each])
        
        # Fill remaining with other types or extra from categories
        remaining = len(results) - len(balanced)
        if remaining > 0:
            extra = (k_type[target_each:] + p_type[target_each:] + other_type)[:remaining]
            balanced.extend(extra)
        
        return balanced
    
    return results


def balance_test_types(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Balance test types in results for multi-domain queries
    If query mentions both technical and behavioral aspects, ensure balanced mix
    """
    query_lower = query.lower()
    
    # Keywords indicating different domains
    technical_keywords = ['java', 'python', 'sql', 'programming', 'coding', 'developer', 'technical', 'software', 'data', 'analyst']
    behavioral_keywords = ['collaborate', 'communication', 'leadership', 'personality', 'teamwork', 'management', 'behavioral', 'soft skills']
    
    has_technical = any(kw in query_lower for kw in technical_keywords)
    has_behavioral = any(kw in query_lower for kw in behavioral_keywords)
    
    # If query has both aspects, balance the results
    if has_technical and has_behavioral:
        k_type = []  # Knowledge & Skills
        p_type = []  # Personality & Behavior
        other_type = []
        
        for result in results:
            test_type = result.get("test_type", "")
            if 'K' in test_type or 'knowledge' in str(test_type).lower():
                k_type.append(result)
            elif 'P' in test_type or 'personality' in str(test_type).lower():
                p_type.append(result)
            else:
                other_type.append(result)
        
        # Aim for 50-50 split or close to it
        target_each = len(results) // 2
        balanced = []
        balanced.extend(k_type[:target_each])
        balanced.extend(p_type[:target_each])
        balanced.extend(other_type[:max(0, len(results) - len(balanced))])
        
        # If we don't have enough of one type, fill with the other
        if len(balanced) < len(results):
            remaining = [r for r in results if r not in balanced]
            balanced.extend(remaining[:len(results) - len(balanced)])
        
        return balanced[:len(results)]
    
    return results


def search_assessments(query: str, max_results: int = 10) -> str:
    """
    Search for SHL assessments based on query with intelligent balancing
    
    This tool searches through 348+ SHL assessments using semantic search
    to find the most relevant assessments for the given query. It automatically
    balances results when the query spans multiple domains (e.g., technical + behavioral).
    
    Args:
        query: Search query describing requirements (e.g., "Data Science assessments")
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        JSON string containing list of relevant assessments with details
    """
    try:
        searcher = get_searcher()
        # Get more results initially for better balancing
        results = searcher.search(query, k=max_results * 2)
        
        # Apply intelligent balancing
        results = balance_test_types(results, query)
        
        # Limit to requested number
        results = results[:max_results]
        
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
