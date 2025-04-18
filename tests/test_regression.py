"""Test suite for regression analysis"""
import pandas as pd
import numpy as np
import pytest
from regression_analysis.regression import perform_regression

def create_correlated_feature(target, p_pos_if_1, p_pos_if_0):
    """Create a feature correlated with the target

    Args:
        target (np.ndarray): Target array of 0s and 1s
        p_pos_if_1 (float): Probability of 1 if target is 1
        p_pos_if_0 (float): Probability of 1 if target is 0
    """
    return np.where(
        target == 1,
        np.random.choice([0, 1], len(target), p=[1-p_pos_if_1, p_pos_if_1]),
        np.random.choice([0, 1], len(target), p=[1-p_pos_if_0, p_pos_if_0])
    )

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100

    # Create correlated sample features
    pregnant = np.random.randint(0, 2, n_samples)
    data = {
        'Folic Acid': create_correlated_feature(pregnant, 0.2, 0.8),
        'Birth Control': create_correlated_feature(pregnant, 0.8, 0.4),
        'Prenatal Vitamins': create_correlated_feature(pregnant, 0.1, 0.9),
        'Wine': create_correlated_feature(pregnant, 0.9, 0.3),
        'PREGNANT': pregnant
    }
    return pd.DataFrame(data)

# pylint: disable=redefined-outer-name
def test_model_accuracy(sample_data, tmp_path):
    """Test if model achieves reasonable accuracy with realistic data"""
    # Save sample data to temp Excel file
    excel_path = tmp_path / "test_data.xlsx"
    sample_data.to_excel(excel_path, index=False)

    # Run regression
    result = perform_regression(excel_path, 'PREGNANT')

    # Check if accuracy is above random chance (0.5)
    assert result['accuracy'] > 0.5

def test_feature_importance(sample_data, tmp_path):  # pylint: disable=redefined-outer-name
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
