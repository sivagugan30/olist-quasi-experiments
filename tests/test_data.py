"""
Tests for data loading and preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataLoader:
    """Tests for data loader functions."""
    
    def test_olist_tables_defined(self):
        """Test that OLIST_TABLES contains expected tables."""
        from src.data.loader import OLIST_TABLES
        
        expected_tables = [
            'orders', 'order_items', 'order_payments', 
            'order_reviews', 'customers', 'products',
            'sellers', 'geolocation', 'category_translation'
        ]
        
        for table in expected_tables:
            assert table in OLIST_TABLES, f"Missing table: {table}"
    
    def test_date_columns_defined(self):
        """Test that DATE_COLUMNS is properly defined."""
        from src.data.loader import DATE_COLUMNS
        
        assert 'orders' in DATE_COLUMNS
        assert 'order_purchase_timestamp' in DATE_COLUMNS['orders']


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_preprocess_orders_adds_columns(self):
        """Test that preprocess_orders adds expected columns."""
        from src.data.preprocessing import preprocess_orders
        
        # Create minimal test dataframe
        orders = pd.DataFrame({
            'order_id': ['1', '2'],
            'order_status': ['delivered', 'canceled'],
            'order_purchase_timestamp': pd.to_datetime(['2018-01-01', '2018-06-01']),
            'order_approved_at': pd.to_datetime(['2018-01-01', '2018-06-01']),
            'order_delivered_carrier_date': pd.to_datetime(['2018-01-02', '2018-06-02']),
            'order_delivered_customer_date': pd.to_datetime(['2018-01-05', '2018-06-15']),
            'order_estimated_delivery_date': pd.to_datetime(['2018-01-10', '2018-06-10']),
        })
        
        result = preprocess_orders(orders)
        
        # Check new columns exist
        expected_cols = [
            'purchase_date', 'purchase_year', 'purchase_month',
            'delivery_time_actual', 'delivery_time_estimated',
            'is_late', 'days_from_deadline', 'is_delivered', 'is_canceled',
            'during_strike', 'post_strike'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_is_late_calculation(self):
        """Test that is_late is calculated correctly."""
        from src.data.preprocessing import preprocess_orders
        
        orders = pd.DataFrame({
            'order_id': ['early', 'late'],
            'order_status': ['delivered', 'delivered'],
            'order_purchase_timestamp': pd.to_datetime(['2018-01-01', '2018-01-01']),
            'order_approved_at': pd.to_datetime(['2018-01-01', '2018-01-01']),
            'order_delivered_carrier_date': pd.to_datetime(['2018-01-02', '2018-01-02']),
            'order_delivered_customer_date': pd.to_datetime(['2018-01-05', '2018-01-15']),
            'order_estimated_delivery_date': pd.to_datetime(['2018-01-10', '2018-01-10']),
        })
        
        result = preprocess_orders(orders)
        
        assert result.loc[result['order_id'] == 'early', 'is_late'].values[0] == False
        assert result.loc[result['order_id'] == 'late', 'is_late'].values[0] == True
    
    def test_aggregate_order_items(self):
        """Test order items aggregation."""
        from src.data.preprocessing import aggregate_order_items
        
        order_items = pd.DataFrame({
            'order_id': ['1', '1', '2'],
            'order_item_id': [1, 2, 1],
            'product_id': ['p1', 'p2', 'p3'],
            'seller_id': ['s1', 's1', 's2'],
            'price': [100.0, 50.0, 200.0],
            'freight_value': [10.0, 5.0, 20.0],
        })
        
        result = aggregate_order_items(order_items)
        
        assert len(result) == 2
        assert result.loc[result['order_id'] == '1', 'n_items'].values[0] == 2
        assert result.loc[result['order_id'] == '1', 'total_price'].values[0] == 150.0
        assert result.loc[result['order_id'] == '2', 'total_value'].values[0] == 220.0


class TestAnalysis:
    """Tests for analysis functions."""
    
    def test_rd_result_to_dict(self):
        """Test RDResult conversion to dictionary."""
        from src.analysis.rd import RDResult
        
        result = RDResult(
            estimate=0.5,
            se=0.1,
            ci_low=0.3,
            ci_high=0.7,
            pvalue=0.001,
            bandwidth=5.0,
            n_left=100,
            n_right=100,
            n_effective=200,
            method='local_polynomial',
            polynomial_order=1,
            kernel='triangular',
        )
        
        d = result.to_dict()
        
        assert d['estimate'] == 0.5
        assert d['pvalue'] == 0.001
        assert 'bandwidth' in d
    
    def test_did_result_to_dict(self):
        """Test DiDResult conversion to dictionary."""
        from src.analysis.did import DiDResult
        
        result = DiDResult(
            estimate=1.5,
            se=0.2,
            ci_low=1.1,
            ci_high=1.9,
            pvalue=0.001,
            n_treatment_pre=50,
            n_treatment_post=50,
            n_control_pre=100,
            n_control_post=100,
            n_total=300,
            method='simple_did',
            covariates=None,
        )
        
        d = result.to_dict()
        
        assert d['estimate'] == 1.5
        assert d['n_total'] == 300


class TestVisualization:
    """Tests for visualization utilities."""
    
    def test_olist_colors_defined(self):
        """Test that OLIST_COLORS contains expected colors."""
        from src.visualization import OLIST_COLORS
        
        expected_colors = ['primary', 'secondary', 'success', 'danger', 
                          'treatment', 'control']
        
        for color in expected_colors:
            assert color in OLIST_COLORS, f"Missing color: {color}"
    
    def test_figure_config_defined(self):
        """Test that FIGURE_CONFIG contains expected settings."""
        from src.visualization import FIGURE_CONFIG
        
        assert 'width' in FIGURE_CONFIG
        assert 'height' in FIGURE_CONFIG
        assert FIGURE_CONFIG['width'] > 0
        assert FIGURE_CONFIG['height'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
