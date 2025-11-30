# 📊 Data-to-Insights RAG Agent

Enterprise Analytics Assistant powered by Groq, LangChain, and Streamlit. Upload your CSV/Excel files and get instant AI-powered insights, natural language queries, and comprehensive data analysis.

![Python Version](https://img.shields.io/badge/python-3.10.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## ✨ Features

- **📤 Smart Data Upload**: Support for CSV and Excel files with automatic data cleaning
- **🤖 RAG-Powered Q&A**: Ask questions about your data in natural language
- **🧹 Automatic Data Cleaning**: Handle missing values, duplicates, outliers, and type conversions
- **💡 AI-Generated Insights**: Get actionable business intelligence recommendations
- **📈 Interactive Visualizations**: Distributions, correlations, and time series analysis
- **💬 Chat Interface**: Conversational analytics with chat history
- **📊 Data Quality Reports**: Comprehensive data health checks

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │ (Frontend)
└────────┬────────┘
         │
         │ REST API
         ▼
┌─────────────────┐
│   FastAPI       │ (Backend)
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬─────────────┐
    │          │          │             │
    ▼          ▼          ▼             ▼
┌─────────┐ ┌──────┐ ┌─────────┐ ┌──────────┐
│ ChromaDB│ │ Groq │ │HuggingFace│ │Cleaning │
│ Vector  │ │ LLM  │ │Embeddings │ │ Module  │
│  Store  │ │      │ │           │ │         │
└─────────┘ └──────┘ └─────────┘ └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10.14 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd data-to-insights-rag-agent
```

2. **Set up the backend**
```bash
cd backend
pip install -r requirements.txt
```

3. **Set up the frontend**
```bash
cd ../frontend
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cd ../backend
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_TEMPERATURE=0.0
CHROMA_PERSIST_DIR=./chroma_db
```

### Running the Application

1. **Start the backend server** (Terminal 1)
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the frontend** (Terminal 2)
```bash
cd frontend
streamlit run main.py
```

3. **Access the application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Usage Guide

### 1. Upload Your Data
- Click "Browse files" in the sidebar
- Select a CSV or Excel file
- Click "🚀 Process Data"
- Review the automatic data cleaning report

### 2. Explore Your Data
Navigate through the tabs:

**📊 Data Overview**
- View data shape, missing values, and column types
- See data preview and column details
- Review the data cleaning report

**💬 Chat & Query**
- Ask natural language questions
- Use suggested questions or type your own
- View chat history

**💡 Insights**
- Click "Generate Insights" for AI analysis
- Get actionable recommendations with confidence levels
- Download insights report

**📈 Visualizations**
- Interactive distributions and histograms
- Correlation heatmaps
- Time series analysis

### Example Questions

```
- "What are the main trends in the data?"
- "Show me summary statistics for sales"
- "Which products have the highest revenue?"
- "What columns have missing data?"
- "Identify any anomalies or outliers"
```

## 🛠️ Technical Stack

### Backend
- **FastAPI**: REST API framework
- **LangChain**: RAG orchestration and LLM integration
- **Groq**: Lightning-fast LLM inference (Llama-3.3-70B)
- **ChromaDB**: Vector database for semantic search
- **HuggingFace**: Free embeddings (all-MiniLM-L6-v2)
- **Pandas**: Data manipulation and analysis

### Frontend
- **Streamlit**: Interactive web application
- **Plotly**: Interactive visualizations
- **Requests**: API communication

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── ingest.py            # Data ingestion pipeline
│   ├── qa.py                # RAG query processing
│   ├── cleaning.py          # Data cleaning module
│   ├── vectorstore.py       # ChromaDB configuration
│   ├── utils.py             # Helper functions
│   ├── requirements.txt     # Backend dependencies
│   └── .env.example         # Environment template
├── frontend/
│   ├── main.py              # Streamlit application
│   └── requirements.txt     # Frontend dependencies
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required |
| `LLM_TEMPERATURE` | Model temperature (0.0-1.0) | 0.0 |
| `CHROMA_PERSIST_DIR` | Vector store location | ./chroma_db |

### Data Cleaning Options

Automatic cleaning handles:
- ✅ Missing values (median/mode imputation or column removal)
- ✅ Duplicate rows removal
- ✅ Data type conversion (dates, numbers)
- ✅ Column name standardization
- ⚠️ Outlier handling (optional, can be enabled)

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and info |
| `/health` | GET | Health check |
| `/ingest` | POST | Upload and process data |
| `/data-quality` | POST | Get data quality report |
| `/query` | POST | Ask questions about data |
| `/insights` | POST | Generate AI insights |

## 📊 Performance Tips

1. **Adjust `k` value**: Lower values (1-2) for faster queries with fewer tokens
2. **Batch size**: Default 20 rows per passage works well for most datasets
3. **Data cleaning**: Disable outlier handling for faster processing
4. **Large files**: Consider sampling or aggregating before upload

## 🐛 Troubleshooting

### Common Issues

**"Request timed out"**
- Reduce dataset size or increase timeout values
- Check network connection to Groq API

**"Out of memory"**
- Reduce `max_rows_per_passage` in `utils.py`
- Process smaller file chunks

**"ChromaDB errors"**
- Delete `chroma_db/` folder and restart
- Ensure proper permissions

**"Dollar signs rendering as LaTeX"**
- Already fixed in code with `answer.replace('$', '\\$')`


## 🙏 Acknowledgments

- **Groq** for blazing-fast LLM inference
- **LangChain** for RAG framework
- **Streamlit** for the amazing UI framework
- **HuggingFace** for free embeddings


---

**Built with ❤️ using Streamlit, LangChain & Groq**
