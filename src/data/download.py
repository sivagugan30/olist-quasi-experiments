"""
Download Olist dataset from Kaggle.

Uses kagglehub for easy downloading without manual credential setup.
"""

import os
import shutil
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/data/ to project root
    return Path(__file__).parent.parent.parent


def get_data_path(subdir: str = "raw") -> Path:
    """
    Get the path to a data subdirectory.
    
    Args:
        subdir: Subdirectory name ('raw' or 'processed')
    
    Returns:
        Path to the data subdirectory
    """
    data_path = get_project_root() / "data" / subdir
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


# Expected files in the Olist dataset
EXPECTED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


def download_olist_data(force: bool = False) -> Path:
    """
    Download Olist Brazilian E-Commerce dataset from Kaggle using kagglehub.
    
    Args:
        force: If True, download even if data already exists
    
    Returns:
        Path to the raw data directory
    
    Raises:
        ImportError: If kagglehub package is not installed
    """
    raw_path = get_data_path("raw")
    
    # Check if data already exists
    existing_files = [f for f in EXPECTED_FILES if (raw_path / f).exists()]
    
    if len(existing_files) == len(EXPECTED_FILES) and not force:
        print(f"[OK] All {len(EXPECTED_FILES)} data files already exist in {raw_path}")
        return raw_path
    
    print(f"Downloading Olist dataset...")
    
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub package not installed. Run: pip install kagglehub"
        )
    
    # Download using kagglehub (handles auth automatically)
    download_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    download_path = Path(download_path)
    
    print(f"Downloaded to: {download_path}")
    print(f"Copying files to {raw_path}...")
    
    # Copy files to our data/raw directory
    for file in download_path.glob("*.csv"):
        dest = raw_path / file.name
        if not dest.exists() or force:
            shutil.copy2(file, dest)
            print(f"  Copied: {file.name}")
    
    # Verify download
    missing = [f for f in EXPECTED_FILES if not (raw_path / f).exists()]
    if missing:
        print(f"⚠ Warning: Missing files after download: {missing}")
    else:
        print(f"[OK] Successfully downloaded {len(EXPECTED_FILES)} files to {raw_path}")
    
    return raw_path


def main():
    """CLI entry point for downloading data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Olist dataset from Kaggle")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if data exists"
    )
    args = parser.parse_args()
    
    download_olist_data(force=args.force)


if __name__ == "__main__":
    main()
