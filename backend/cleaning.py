import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning for analytics datasets.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_report = {
            "original_shape": df.shape,
            "actions_taken": [],
            "columns_modified": [],
            "rows_removed": 0,
            "missing_handled": 0
        }
    
    def get_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            "shape": self.df.shape,
            "total_cells": self.df.shape[0] * self.df.shape[1],
            "missing_values": {},
            "duplicate_rows": 0,
            "data_types": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "issues": []
        }
        
        # Missing values
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            report["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }
            if missing_pct > 50:
                report["issues"].append(f"{col}: {missing_pct:.1f}% missing (High!)")
        
        # Duplicates
        report["duplicate_rows"] = int(self.df.duplicated().sum())
        if report["duplicate_rows"] > 0:
            report["issues"].append(f"Found {report['duplicate_rows']} duplicate rows")
        
        # Data types
        for col, dtype in self.df.dtypes.items():
            dtype_str = str(dtype)
            report["data_types"][col] = dtype_str
            
            if "int" in dtype_str or "float" in dtype_str:
                report["numeric_columns"].append(col)
            elif "datetime" in dtype_str:
                report["date_columns"].append(col)
            else:
                report["categorical_columns"].append(col)
        
        return report
    
    def clean_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Handle missing values based on strategy.
        
        Strategies:
        - auto: Smart handling based on data type and missing %
        - drop_rows: Remove rows with any missing values
        - drop_cols: Remove columns with >50% missing
        - fill_mean: Fill numeric with mean
        - fill_median: Fill numeric with median
        - fill_mode: Fill categorical with mode
        """
        initial_missing = self.df.isnull().sum().sum()
        
        if strategy == "auto":
            for col in self.df.columns:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                
                # Drop column if >70% missing
                if missing_pct > 70:
                    self.df = self.df.drop(columns=[col])
                    self.cleaning_report["actions_taken"].append(
                        f"Dropped column '{col}' ({missing_pct:.1f}% missing)"
                    )
                    continue
                
                # Fill missing values
                if missing_pct > 0:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        # Use median for numeric
                        fill_value = self.df[col].median()
                        self.df[col].fillna(fill_value, inplace=True)
                        self.cleaning_report["actions_taken"].append(
                            f"Filled {col} with median ({fill_value})"
                        )
                    else:
                        # Use mode for categorical
                        fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                        self.df[col].fillna(fill_value, inplace=True)
                        self.cleaning_report["actions_taken"].append(
                            f"Filled {col} with mode ('{fill_value}')"
                        )
        
        elif strategy == "drop_rows":
            before = len(self.df)
            self.df = self.df.dropna()
            removed = before - len(self.df)
            self.cleaning_report["rows_removed"] += removed
            self.cleaning_report["actions_taken"].append(
                f"Removed {removed} rows with missing values"
            )
        
        elif strategy == "drop_cols":
            for col in self.df.columns:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                if missing_pct > 50:
                    self.df = self.df.drop(columns=[col])
                    self.cleaning_report["actions_taken"].append(
                        f"Dropped column '{col}' ({missing_pct:.1f}% missing)"
                    )
        
        final_missing = self.df.isnull().sum().sum()
        self.cleaning_report["missing_handled"] = int(initial_missing - final_missing)
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        
        if removed > 0:
            self.cleaning_report["rows_removed"] += removed
            self.cleaning_report["actions_taken"].append(
                f"Removed {removed} duplicate rows"
            )
        
        return self.df
    
    def handle_outliers(self, columns: List[str] = None, method: str = "iqr") -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Methods:
        - iqr: Interquartile range (remove values outside 1.5*IQR)
        - zscore: Remove values with z-score > 3
        - cap: Cap values at 99th percentile
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers_before = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                
                if outliers_before > 0:
                    self.cleaning_report["actions_taken"].append(
                        f"Removed {outliers_before} outliers from '{col}' (IQR method)"
                    )
            
            elif method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_before = (z_scores > 3).sum()
                self.df = self.df[z_scores <= 3]
                
                if outliers_before > 0:
                    self.cleaning_report["actions_taken"].append(
                        f"Removed {outliers_before} outliers from '{col}' (Z-score method)"
                    )
            
            elif method == "cap":
                p99 = self.df[col].quantile(0.99)
                p1 = self.df[col].quantile(0.01)
                capped = ((self.df[col] > p99) | (self.df[col] < p1)).sum()
                self.df[col] = self.df[col].clip(lower=p1, upper=p99)
                
                if capped > 0:
                    self.cleaning_report["actions_taken"].append(
                        f"Capped {capped} values in '{col}' to 1st-99th percentile"
                    )
        
        rows_removed = initial_rows - len(self.df)
        self.cleaning_report["rows_removed"] += rows_removed
        
        return self.df
    
    def convert_data_types(self) -> pd.DataFrame:
        """Auto-detect and convert appropriate data types."""
        for col in self.df.columns:
            # Try to convert to datetime
            if self.df[col].dtype == 'object':
                try:
                    # Check if it looks like a date
                    sample = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else ""
                    if any(char in str(sample) for char in ['/', '-', ':']):
                        # Try to convert - use try/except instead of errors='ignore'
                        try:
                            converted = pd.to_datetime(self.df[col])
                            self.df[col] = converted
                            self.cleaning_report["actions_taken"].append(
                                f"Converted '{col}' to datetime"
                            )
                        except (ValueError, TypeError):
                            # If conversion fails, leave as is
                            pass
                except:
                    pass
            
            # Try to convert to numeric
            if self.df[col].dtype == 'object':
                try:
                    numeric_col = pd.to_numeric(self.df[col], errors='coerce')
                    # If >50% successfully converted, keep it
                    if numeric_col.notna().sum() / len(self.df) > 0.5:
                        self.df[col] = numeric_col
                        self.cleaning_report["actions_taken"].append(
                            f"Converted '{col}' to numeric"
                        )
                except:
                    pass
        
        return self.df
    
    def standardize_column_names(self) -> pd.DataFrame:
        """Standardize column names (lowercase, no spaces)."""
        old_names = self.df.columns.tolist()
        new_names = [
            col.lower().strip().replace(' ', '_').replace('-', '_')
            for col in self.df.columns
        ]
        
        if old_names != new_names:
            self.df.columns = new_names
            self.cleaning_report["actions_taken"].append(
                "Standardized column names (lowercase, underscores)"
            )
        
        return self.df
    
    def auto_clean(self, 
                   handle_missing: bool = True,
                   remove_duplicates: bool = True,
                   handle_outliers: bool = False,
                   convert_types: bool = True,
                   standardize_names: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform automatic comprehensive data cleaning.
        
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        logger.info("Starting automatic data cleaning...")
        
        # Standardize column names first
        if standardize_names:
            self.standardize_column_names()
        
        # Convert data types
        if convert_types:
            self.convert_data_types()
        
        # Remove duplicates
        if remove_duplicates:
            self.remove_duplicates()
        
        # Handle missing values
        if handle_missing:
            self.clean_missing_values(strategy="auto")
        
        # Handle outliers (optional, can be aggressive)
        if handle_outliers:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                self.handle_outliers(columns=numeric_cols, method="cap")
        
        # Update final report
        self.cleaning_report["final_shape"] = self.df.shape
        self.cleaning_report["total_rows_removed"] = (
            self.original_df.shape[0] - self.df.shape[0]
        )
        self.cleaning_report["total_columns_removed"] = (
            self.original_df.shape[1] - self.df.shape[1]
        )
        
        logger.info(f"Cleaning complete: {len(self.cleaning_report['actions_taken'])} actions taken")
        
        return self.df, self.cleaning_report
    
    def get_cleaning_summary(self) -> str:
        """Get a human-readable summary of cleaning actions."""
        summary = []
        summary.append(f"📊 Original Data: {self.cleaning_report['original_shape'][0]} rows × {self.cleaning_report['original_shape'][1]} columns")
        summary.append(f"📊 Cleaned Data: {self.cleaning_report['final_shape'][0]} rows × {self.cleaning_report['final_shape'][1]} columns")
        summary.append("")
        summary.append(f"✅ Actions Performed: {len(self.cleaning_report['actions_taken'])}")
        summary.append("")
        
        for action in self.cleaning_report['actions_taken']:
            summary.append(f"  • {action}")
        
        summary.append("")
        summary.append(f"🗑️ Total Rows Removed: {self.cleaning_report['total_rows_removed']}")
        summary.append(f"🗑️ Total Columns Removed: {self.cleaning_report['total_columns_removed']}")
        summary.append(f"✨ Missing Values Handled: {self.cleaning_report['missing_handled']}")
        
        return "\n".join(summary)


def clean_dataframe(df: pd.DataFrame, 
                    auto: bool = True,
                    **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to clean a dataframe.
    
    Args:
        df: Input dataframe
        auto: If True, perform automatic cleaning
        **kwargs: Additional arguments for auto_clean()
    
    Returns:
        Tuple of (cleaned_df, report)
    """
    cleaner = DataCleaner(df)
    
    if auto:
        return cleaner.auto_clean(**kwargs)
    else:
        return df, cleaner.get_data_quality_report()