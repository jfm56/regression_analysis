"""Test suite for regression analysis"""
import pandas as pd
import numpy as np
import pytest
from regression import perform_regression

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features
    data = {
        'Folic Acid': np.random.randint(0, 2, n_samples),
        'Birth Control': np.random.randint(0, 2, n_samples),
        'Prenatal Vitamins': np.random.randint(0, 2, n_samples),
        'Wine': np.random.randint(0, 2, n_samples),
        'PREGNANT': np.random.randint(0, 2, n_samples)
    }
    return pd.DataFrame(data)

def test_model_accuracy(sample_data, tmp_path):
    """Test if model achieves reasonable accuracy"""
    # Save sample data to temp Excel file
    excel_path = tmp_path / "test_data.xlsx"
    sample_data.to_excel(excel_path, index=False)
    
    # Run regression
    result = perform_regression(excel_path, 'PREGNANT')
    
    # Check if accuracy is above random chance (0.5)
    assert result['accuracy'] > 0.5

def test_feature_importance(sample_data, tmp_path):
    """Test if feature importance values are reasonable"""
    excel_path = tmp_path / "test_data.xlsx"
    sample_data.to_excel(excel_path, index=False)
    
    result = perform_regression(excel_path, 'PREGNANT')
    
    # Check if we have importance values for all features
    assert len(result['feature_importance']) == len(sample_data.columns) - 1

def test_invalid_input():
    """Test if function handles invalid input appropriately"""
    with pytest.raises(FileNotFoundError):
        perform_regression("nonexistent.xlsx", "PREGNANT")
