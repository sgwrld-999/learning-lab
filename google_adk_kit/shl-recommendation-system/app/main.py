"""
SHL Assessment Recommendation System
Main FastAPI application with direct FAISS search integration
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shl_agent.tools.search_tool import search_assessments

# Load environment variables
load_dotenv()

APP_NAME = "SHL Assessment Recommender"

# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="GenAI Assessment Recommendation System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Request/Response Models
class RecommendRequest(BaseModel):
    query: str


class AssessmentRecommendation(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[str]
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentRecommendation]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint as per SHL requirements"""
    return {"status": "healthy"}


# Main recommendation endpoint
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Assessment Recommendation Endpoint
    Returns 5-10 recommended assessments based on query
    """
    try:
        # Search using FAISS
        results_json = search_assessments(request.query, max_results=10)
        results = json.loads(results_json)
        
        # Format recommendations
        recommendations = []
        for result in results:
            # Handle test_type - ensure it's a list
            test_type = result.get("test_type", [])
            if isinstance(test_type, str):
                test_type = [test_type] if test_type else []
            
            recommendations.append(AssessmentRecommendation(
                url=result.get("url", ""),
                name=result.get("title", ""),
                adaptive_support=result.get("adaptive", "No") if result.get("adaptive") else "No",
                description=result.get("description", ""),
                duration=result.get("duration", ""),
                remote_support=result.get("remote_testing", "Check details"),
                test_type=test_type
            ))
        
        return RecommendResponse(recommended_assessments=recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint - serve chatbot interface
@app.get("/")
async def root():
    """Serve chatbot frontend"""
    chat_path = static_path / "chat.html"
    if chat_path.exists():
        return FileResponse(chat_path)
    return {
        "service": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "recommend": "POST /recommend",
            "chat": "GET / (chatbot interface)",
            "table": "GET /table (table view)"
        }
    }


# Table view endpoint
@app.get("/table")
async def table_view():
    """Serve table view frontend"""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Table view not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
