"""
Format tool for structuring assessment recommendations
"""

import json
from typing import List, Dict, Any


def format_recommendations(assessments_json: str) -> str:
    """
    Format assessment recommendations into structured response
    
    This tool takes search results and formats them into the required
    API response structure for SHL recommendations.
    
    Args:
        assessments_json: JSON string of assessment results from search
        
    Returns:
        JSON string with formatted recommendations matching API spec
    """
    try:
        assessments = json.loads(assessments_json)
        
        # Format each assessment
        formatted = []
        for assessment in assessments:
            formatted.append({
                "url": assessment.get("url", ""),
                "name": assessment.get("title", ""),
                "adaptive_support": assessment.get("adaptive", "No"),
                "description": assessment.get("description", ""),
                "duration": assessment.get("duration"),
                "remote_support": assessment.get("remote_testing", "No"),
                "test_type": assessment.get("test_type", [])
            })
        
        return json.dumps({
            "recommended_assessments": formatted
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "recommended_assessments": [],
            "error": str(e)
        })
