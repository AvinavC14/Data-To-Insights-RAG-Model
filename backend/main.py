import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.ingest import ingest_dataframe
from backend.utils import read_uploaded_file_bytes, get_dataframe_summary
from backend.qa import run_query, generate_insights
from backend.cleaning import clean_dataframe, DataCleaner  # NEW
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Data-to-Insights RAG Agent",
    description="Enterprise Analytics Assistant with RAG powered by Groq",
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

class QueryRequest(BaseModel):
    question: str
    k: int = 4

class InsightsRequest(BaseModel):
    df_summary: dict
    k: int = 6

class CleaningRequest(BaseModel):
    handle_missing: bool = True
    remove_duplicates: bool = True
    handle_outliers: bool = False
    convert_types: bool = True
    standardize_names: bool = True

@app.get("/")
def root():
    return {
        "message": "Data-to-Insights RAG Agent API",
        "version": "1.0.0",
        "status": "running",
        "powered_by": "Groq + LangChain",
        "features": [
            "CSV/Excel Upload",
            "Automatic Data Cleaning",  # NEW
            "Data Quality Reports",     # NEW
            "RAG-powered Q&A",
            "AI Insights Generation",
            "Interactive Visualizations"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), auto_clean: bool = True):
    """
    Ingest a CSV or Excel file into the vector database.
    Now includes automatic data cleaning!
    
    Args:
        file: The uploaded file
        auto_clean: Whether to automatically clean the data (default: True)
    
    Returns:
        - success: bool
        - filename: str
        - ingest_result: dict with documents_ingested and rows_processed
        - data_summary: dict with shape, columns, dtypes, missing values, statistics
        - cleaning_report: dict with cleaning actions taken (if auto_clean=True)
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read file
        contents = await file.read()
        df = read_uploaded_file_bytes(contents, file.filename)
        
        logger.info(f"File parsed successfully. Original shape: {df.shape}")
        
        # NEW: Data Cleaning
        cleaning_report = None
        if auto_clean:
            logger.info("Starting automatic data cleaning...")
            cleaner = DataCleaner(df)
            df, cleaning_report = cleaner.auto_clean(
                handle_missing=True,
                remove_duplicates=True,
                handle_outliers=False,  # Conservative by default
                convert_types=True,
                standardize_names=True
            )
            logger.info(f"Cleaning complete. New shape: {df.shape}")
            logger.info(f"Cleaning summary:\n{cleaner.get_cleaning_summary()}")
        
        # Get summary of cleaned data
        summary = get_dataframe_summary(df)
        
        # Ingest into vector DB
        result = ingest_dataframe(df, clear_existing=True)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Ingestion failed"))
        
        logger.info(f"Ingestion completed: {result}")
        
        response = {
            "success": True,
            "filename": file.filename,
            "ingest_result": result,
            "data_summary": summary
        }
        
        # Add cleaning report if cleaning was performed
        if cleaning_report:
            response["cleaning_report"] = cleaning_report
            response["cleaning_summary"] = cleaner.get_cleaning_summary()
        
        return response
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/data-quality")
async def check_data_quality(file: UploadFile = File(...)):
    """
    Check data quality without ingestion.
    Returns a comprehensive data quality report.
    """
    try:
        logger.info(f"Checking data quality for: {file.filename}")
        
        # Read file
        contents = await file.read()
        df = read_uploaded_file_bytes(contents, file.filename)
        
        # Generate quality report
        cleaner = DataCleaner(df)
        quality_report = cleaner.get_data_quality_report()
        
        return {
            "success": True,
            "filename": file.filename,
            "quality_report": quality_report
        }
        
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
def query(req: QueryRequest):
    """
    Query the ingested data using RAG.
    
    Args:
        question: str - The question to ask
        k: int - Number of relevant documents to retrieve (default: 4)
    
    Returns:
        - success: bool
        - question: str
        - answer: str
    """
    try:
        logger.info(f"Received query: {req.question}")
        
        answer = run_query(req.question, k=req.k)
        
        logger.info(f"Query answered successfully")
        
        return {
            "success": True,
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insights")
def insights(req: InsightsRequest):
    """
    Generate business insights from data summary.
    
    Args:
        df_summary: dict - DataFrame summary with statistics
        k: int - Number of insights to generate (default: 6)
    
    Returns:
        - success: bool
        - insights: str - Formatted insights
    """
    try:
        logger.info(f"Generating {req.k} insights")
        
        insights_text = generate_insights(req.df_summary, k=req.k)
        
        logger.info("Insights generated successfully")
        
        return {
            "success": True,
            "insights": insights_text
        }
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=600,
        limit_concurrency=10,
        limit_max_requests=1000
    )