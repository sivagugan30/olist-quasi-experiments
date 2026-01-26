"""
Data loading utilities for Olist dataset.

Provides functions to load individual tables or all tables at once.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .download import get_data_path


# Define the Olist dataset tables
OLIST_TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Date columns that should be parsed
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def load_raw_data(
    table_name: str,
    data_path: Optional[Path] = None,
    parse_dates: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load a single table from the Olist dataset.
    
    Args:
        table_name: Name of the table (e.g., 'orders', 'order_items')
        data_path: Path to raw data directory. If None, uses default.
        parse_dates: Whether to parse date columns automatically
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame with the requested table
    
    Raises:
        ValueError: If table_name is not valid
        FileNotFoundError: If the data file doesn't exist
    """
    if table_name not in OLIST_TABLES:
        valid_names = list(OLIST_TABLES.keys())
        raise ValueError(
            f"Unknown table: {table_name}. Valid options: {valid_names}"
        )
    
    if data_path is None:
        data_path = get_data_path("raw")
    
    file_path = data_path / OLIST_TABLES[table_name]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            "Run: python -m src.data.download to download the dataset."
        )
    
    # Set up date parsing if requested
    read_kwargs = kwargs.copy()
    if parse_dates and table_name in DATE_COLUMNS:
        read_kwargs["parse_dates"] = DATE_COLUMNS[table_name]
    
    df = pd.read_csv(file_path, **read_kwargs)
    
    return df


def load_all_tables(
    data_path: Optional[Path] = None,
    parse_dates: bool = True,
    exclude: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all Olist tables into a dictionary.
    
    Args:
        data_path: Path to raw data directory. If None, uses default.
        parse_dates: Whether to parse date columns automatically
        exclude: List of table names to exclude (e.g., ['geolocation'] for memory)
    
    Returns:
        Dictionary mapping table names to DataFrames
    """
    exclude = exclude or []
    
    tables = {}
    for table_name in OLIST_TABLES:
        if table_name in exclude:
            continue
        
        try:
            tables[table_name] = load_raw_data(
                table_name,
                data_path=data_path,
                parse_dates=parse_dates
            )
            print(f"[OK] Loaded {table_name}: {len(tables[table_name]):,} rows")
        except FileNotFoundError as e:
            print(f"✗ Failed to load {table_name}: {e}")
    
    return tables


def get_table_info(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get summary information about all loaded tables.
    
    Args:
        tables: Dictionary of DataFrames
    
    Returns:
        DataFrame with table statistics
    """
    info = []
    for name, df in tables.items():
        info.append({
            "table": name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_names": ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else ""),
        })
    
    return pd.DataFrame(info)
