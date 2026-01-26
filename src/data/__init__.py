"""
Data loading and preprocessing utilities for Olist dataset.
"""

from .download import (
    download_olist_data,
    get_project_root,
    get_data_path,
)

from .loader import (
    load_raw_data,
    load_all_tables,
    get_table_info,
    OLIST_TABLES,
    DATE_COLUMNS,
)

from .preprocessing import (
    preprocess_orders,
    aggregate_order_items,
    aggregate_payments,
    aggregate_reviews,
    merge_order_data,
    create_shipping_threshold_features,
    create_analysis_dataset,
    load_or_create_analysis_dataset,
)

__all__ = [
    # Download
    "download_olist_data",
    "get_project_root",
    "get_data_path",
    # Loader
    "load_raw_data",
    "load_all_tables",
    "get_table_info",
    "OLIST_TABLES",
    "DATE_COLUMNS",
    # Preprocessing
    "preprocess_orders",
    "aggregate_order_items",
    "aggregate_payments",
    "aggregate_reviews",
    "merge_order_data",
    "create_shipping_threshold_features",
    "create_analysis_dataset",
    "load_or_create_analysis_dataset",
]
