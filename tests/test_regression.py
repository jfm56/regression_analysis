"""Test suite for regression analysis"""
import pandas as pd
import numpy as np
import pytest
from regression_analysis.regression import perform_regression

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated sample features
    pregnant = np.random.randint(0, 2, n_samples)
    data = {
        'Folic Acid': np.where(pregnant == 1, np.random.choice([0, 1], n_samples, p=[0.2, 0.8]), np.random.choice([0, 1], n_samples, p=[0.8, 0.2])),
        'Birth Control': np.where(pregnant == 1, np.random.choice([0, 1], n_samples, p=[0.8, 0.2]), np.random.choice([0, 1], n_samples, p=[0.4, 0.6])),
        'Prenatal Vitamins': np.where(pregnant == 1, np.random.choice([0, 1], n_samples, p=[0.1, 0.9]), np.random.choice([0, 1], n_samples, p=[0.9, 0.1])),
        'Wine': np.where(pregnant == 1, np.random.choice([0, 1], n_samples, p=[0.9, 0.1]), np.random.choice([0, 1], n_samples, p=[0.3, 0.7])),
        'PREGNANT': pregnant
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
