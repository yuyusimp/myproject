"""
Script to create a StandardScaler for the strand prediction model.
This ensures consistent feature scaling between training and prediction.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def create_and_save_scaler():
    """
    Create a StandardScaler based on typical grade ranges and save to models folder.
    """
    # Generate synthetic data that represents the range of possible grades (e.g., 65-100)
    # 8 features: Math, Science, English, Filipino, MAPEH, AP, ESP, TLE
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic grade data
    sample_size = 1000
    subjects = 8
    
    # Create random grades between 65 and 100
    synthetic_grades = np.random.uniform(65, 100, (sample_size, subjects))
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(synthetic_grades)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"StandardScaler created and saved to {scaler_path}")
    return scaler_path

if __name__ == "__main__":
    create_and_save_scaler()
