import pandas as pd
import numpy as np
from io import StringIO, BytesIO

def df_to_passages(df, max_rows_per_passage=50):
    """
    Convert DataFrame rows into text passages for RAG.
    Groups multiple rows into passages for better context.
    Larger batches = fewer documents = faster processing.
    """
    passages = []
    total_rows = len(df)
    
    for start_idx in range(0, total_rows, max_rows_per_passage):
        end_idx = min(start_idx + max_rows_per_passage, total_rows)
        chunk_df = df.iloc[start_idx:end_idx]
        
        # Create a readable text representation
        text_parts = [f"Data rows {start_idx} to {end_idx-1}:\n"]
        
        for idx, row in chunk_df.iterrows():
            row_text = f"Row {idx}: "
            items = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    val_str = "NULL"
                else:
                    val_str = str(val)
                items.append(f"{col}={val_str}")
            row_text += ", ".join(items)
            text_parts.append(row_text)
        
        passage_text = "\n".join(text_parts)
        
        metadata = {
            "start_row": int(start_idx),
            "end_row": int(end_idx - 1),
            "num_rows": int(end_idx - start_idx)
        }
        
        passages.append({
            "text": passage_text,
            "metadata": metadata
        })
    
    return passages

def read_uploaded_file_bytes(file_bytes, filename):
    """
    Read uploaded file (CSV or Excel) and return pandas DataFrame.
    """
    try:
        if filename.lower().endswith('.csv'):
            s = StringIO(file_bytes.decode('utf-8'))
            df = pd.read_csv(s)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_bytes))
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        return df
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def sanitize_for_json(obj):
    """
    Recursively sanitize objects for JSON serialization.
    Converts NaN, Infinity, and other non-JSON-compliant values.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (float, int)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    else:
        return obj

def get_dataframe_summary(df):
    """
    Generate a comprehensive summary of the DataFrame.
    All values are sanitized for JSON serialization.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    
    # Get summary statistics for numeric columns
    if len(df.select_dtypes(include='number').columns) > 0:
        stats_df = df.describe()
        # Convert to dict and sanitize
        summary_stats = stats_df.to_dict()
        summary["summary_stats"] = sanitize_for_json(summary_stats)
    else:
        summary["summary_stats"] = {}
    
    # Sanitize the entire summary to ensure JSON compliance
    summary = sanitize_for_json(summary)
    
    return summary