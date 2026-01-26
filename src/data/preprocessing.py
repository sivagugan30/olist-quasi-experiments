"""
Data preprocessing and feature engineering for Olist dataset.

Creates derived features needed for quasi-experimental analyses.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from .download import get_data_path


def preprocess_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the orders table with derived features.
    
    Args:
        orders: Raw orders DataFrame
    
    Returns:
        Orders DataFrame with additional features
    """
    df = orders.copy()
    
    # Ensure datetime columns
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Extract date components
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    
    # Calculate delivery times (in days)
    df["delivery_time_actual"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    
    df["delivery_time_estimated"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    
    df["carrier_handoff_time"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    
    # Late delivery indicator (key for RD analysis)
    df["delivery_delay"] = df["delivery_time_actual"] - df["delivery_time_estimated"]
    df["is_late"] = df["delivery_delay"] > 0
    
    # Days relative to deadline (running variable for RD)
    # Negative = delivered early, Positive = delivered late
    df["days_from_deadline"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    
    # Order status flags
    df["is_delivered"] = df["order_status"] == "delivered"
    df["is_canceled"] = df["order_status"] == "canceled"
    df["is_completed"] = df["order_status"].isin(["delivered", "canceled"])
    
    # Truckers strike period (May 21 - June 1, 2018)
    strike_start = pd.Timestamp("2018-05-21")
    strike_end = pd.Timestamp("2018-06-01")
    
    df["during_strike"] = (
        (df["order_purchase_timestamp"] >= strike_start) &
        (df["order_purchase_timestamp"] <= strike_end)
    )
    
    # Pre/post strike period for DiD
    df["post_strike"] = df["order_purchase_timestamp"] >= strike_start
    
    return df


def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order items to order level.
    
    Args:
        order_items: Raw order items DataFrame
    
    Returns:
        Order-level aggregated metrics
    """
    agg = order_items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()
    
    agg["total_value"] = agg["total_price"] + agg["total_freight"]
    agg["avg_item_price"] = agg["total_price"] / agg["n_items"]
    agg["freight_ratio"] = agg["total_freight"] / agg["total_value"]
    
    return agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate payment data to order level.
    
    Args:
        payments: Raw payments DataFrame
    
    Returns:
        Order-level payment metrics
    """
    # Get primary payment method (highest value)
    primary_payment = (
        payments
        .sort_values("payment_value", ascending=False)
        .groupby("order_id")
        .first()
        .reset_index()[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    
    # Aggregate metrics
    agg = payments.groupby("order_id").agg(
        n_payments=("payment_sequential", "max"),
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        avg_installments=("payment_installments", "mean"),
    ).reset_index()
    
    # Flag for installment usage
    agg["used_installments"] = agg["max_installments"] > 1
    
    # Merge primary payment type
    agg = agg.merge(primary_payment, on="order_id", how="left")
    
    return agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate reviews to order level (handle multiple reviews per order).
    
    Args:
        reviews: Raw reviews DataFrame
    
    Returns:
        Order-level review metrics
    """
    # Take the most recent review per order
    reviews_sorted = reviews.sort_values(
        "review_creation_date", ascending=False
    )
    latest = reviews_sorted.groupby("order_id").first().reset_index()
    
    return latest[[
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
    ]]


def merge_order_data(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all order-related tables into a single analysis dataset.
    
    Args:
        orders: Preprocessed orders DataFrame
        order_items: Raw order items DataFrame
        payments: Raw payments DataFrame
        reviews: Raw reviews DataFrame
        customers: Raw customers DataFrame
    
    Returns:
        Merged order-level dataset
    """
    # Aggregate detailed tables
    items_agg = aggregate_order_items(order_items)
    payments_agg = aggregate_payments(payments)
    reviews_agg = aggregate_reviews(reviews)
    
    # Merge all together
    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")
    df = df.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left"
    )
    
    return df


def create_shipping_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for shipping threshold fuzzy RD analysis.
    
    Common Brazilian e-commerce shipping thresholds: R$99, R$149, R$199
    
    Args:
        df: Merged order DataFrame
    
    Returns:
        DataFrame with shipping threshold features
    """
    df = df.copy()
    
    # Common free shipping thresholds
    thresholds = [99, 149, 199]
    
    for threshold in thresholds:
        # Distance from threshold
        df[f"dist_from_{threshold}"] = df["total_price"] - threshold
        
        # Above threshold indicator
        df[f"above_{threshold}"] = df["total_price"] >= threshold
        
        # Bandwidth indicator (within R$20 of threshold)
        df[f"near_{threshold}"] = abs(df[f"dist_from_{threshold}"]) <= 20
    
    return df


def create_analysis_dataset(
    tables: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create the full analysis dataset from raw tables.
    
    Args:
        tables: Dictionary of raw DataFrames
        save_path: If provided, save the result to this path
    
    Returns:
        Full analysis-ready DataFrame
    """
    # Preprocess orders
    orders = preprocess_orders(tables["orders"])
    
    # Merge all tables
    df = merge_order_data(
        orders=orders,
        order_items=tables["order_items"],
        payments=tables["order_payments"],
        reviews=tables["order_reviews"],
        customers=tables["customers"],
    )
    
    # Add shipping threshold features
    df = create_shipping_threshold_features(df)
    
    # Add product category info (for potential heterogeneity analysis)
    product_cats = tables["products"][["product_id", "product_category_name"]].drop_duplicates()
    
    # Get most common category per order
    items_cats = tables["order_items"].merge(product_cats, on="product_id", how="left")
    order_cats = (
        items_cats.groupby("order_id")["product_category_name"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        .reset_index()
        .rename(columns={"product_category_name": "primary_category"})
    )
    df = df.merge(order_cats, on="order_id", how="left")
    
    # Translate categories
    cat_trans = tables["category_translation"].set_index("product_category_name")[
        "product_category_name_english"
    ].to_dict()
    df["primary_category_english"] = df["primary_category"].map(cat_trans)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"[OK] Saved analysis dataset to {save_path}")
    
    return df


def load_or_create_analysis_dataset(
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load analysis dataset from cache or create it.
    
    Args:
        tables: Dictionary of raw DataFrames (required if force_rebuild or no cache)
        force_rebuild: If True, recreate even if cache exists
    
    Returns:
        Analysis-ready DataFrame
    """
    cache_path = get_data_path("processed") / "analysis_dataset.parquet"
    
    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_parquet(cache_path)
    
    if tables is None:
        raise ValueError(
            "tables must be provided when cache doesn't exist or force_rebuild=True"
        )
    
    return create_analysis_dataset(tables, save_path=cache_path)
