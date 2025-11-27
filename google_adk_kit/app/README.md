# SHL Assessment Recommendation System

AI-powered assessment recommendation system built with Google ADK (Agent Development Kit) and in-memory FAISS vector search.

## 🎯 Features

- **Semantic Search**: Find relevant assessments using natural language queries
- **In-Memory FAISS**: Fast vector search with 348+ SHL assessments
- **ADK-Powered Agent**: Conversational AI using Google's Gemini 2.0 Flash
- **REST API**: Clean endpoints following SHL specifications
- **Beautiful Frontend**: User-friendly interface for querying assessments
- **Real-time Results**: Sub-second search performance

## 🏗️ Architecture

Built following the ADK voice-agent pattern:

```
app/
├── main.py                 # FastAPI application
├── shl_agent/
│   ├── agent.py           # ADK agent definition
│   └── tools/
│       ├── search_tool.py # In-memory FAISS search
│       └── format_tool.py # Response formatting
├── static/
│   └── index.html         # Frontend UI
└── requirements.txt
```

## 📋 Prerequisites

- Python 3.9+
- Google API Key (for Gemini model)
- Virtual environment recommended

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Run the Application

```bash
python main.py
```

The server will start at `http://localhost:8000`

### 4. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Recommendations:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Data Science assessments"}'
```

### 5. Use the Frontend

Open your browser and navigate to:
```
http://localhost:8000
```

## 📚 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy"
}
```

#### `POST /recommend`
Get assessment recommendations

**Request:**
```json
{
  "query": "Python developer assessment"
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/...",
      "name": "Python Programming",
      "adaptive_support": "Yes",
      "description": "Assess Python coding skills...",
      "duration": 30,
      "remote_support": "Yes",
      "test_type": ["K", "A"]
    }
  ]
}
```

## 🔍 Test Type Codes

- **K**: Knowledge & Skills
- **P**: Personality & Behavior  
- **C**: Competencies
- **A**: Abilities (Cognitive)
- **S**: Simulation
- **B**: Behavioral
- **D**: Development
- **E**: Emotional Intelligence

## 🛠️ How It Works

1. **Query Processing**: User submits natural language query
2. **ADK Agent**: Gemini 2.0 analyzes query and determines search strategy
3. **FAISS Search**: In-memory vector search finds relevant assessments
4. **Formatting**: Results formatted to match SHL API specifications
5. **Response**: Structured JSON returned to client

## 📊 Data

The system indexes 348+ SHL assessments from:
```
../data/individual-assessment.json
```

Each assessment includes:
- Title and description
- Duration and test types
- Job levels and requirements
- URLs and remote/adaptive support

## 🎨 Frontend Features

- **Clean UI**: Modern gradient design
- **Example Queries**: Quick-start templates
- **Real-time Search**: Instant results
- **Responsive Cards**: Detailed assessment information
- **Direct Links**: One-click access to assessments

## 🔧 Configuration

Edit `.env` file for customization:

```env
GOOGLE_API_KEY=your_key_here
HOST=0.0.0.0
PORT=8000
DATA_PATH=../data/individual-assessment.json
EMBEDDING_MODEL=all-MiniLM-L6-v2
GEMINI_MODEL=gemini-2.0-flash-exp
DEFAULT_MAX_RESULTS=10
```

## 🧪 Example Queries

Try these queries:

- "Data Science assessments"
- "Leadership and management"
- "Python programming test"
- "Customer service roles"
- "Project manager competencies"
- "Entry level technical assessments"
- "Personality tests for sales"

## 📝 Development

### Project Structure

- **main.py**: FastAPI app with health and recommend endpoints
- **shl_agent/agent.py**: ADK agent configuration
- **shl_agent/tools/search_tool.py**: In-memory FAISS implementation
- **shl_agent/tools/format_tool.py**: Response formatting
- **static/index.html**: Frontend UI

### Adding New Tools

1. Create tool function in `shl_agent/tools/`
2. Import in `shl_agent/agent.py`
3. Add to agent's `tools` list

## 🐛 Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**Google API key missing:**
```bash
# Set in .env file
GOOGLE_API_KEY=your_key_here
```

**Data file not found:**
Ensure `../data/individual-assessment.json` exists relative to app/

**Port already in use:**
Change PORT in .env or run:
```bash
uvicorn main:app --port 8001
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Support

For issues or questions, please open a GitHub issue.

---

Built with ❤️ using Google ADK and FAISS
