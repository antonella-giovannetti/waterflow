import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from experiment import preprocess_data


def test_preprocess_data_removes_nans_and_outliers():
    df = pd.DataFrame({
        "ph": [None, 7.0, 9.0, None],
        "Sulfate": [200.0, None, 250.0, 300.0],
        "Trihalomethanes": [80.0, 100.0, None, 90.0],
        "Potability": [1, 0, 1, 0],
        "Hardness": [150, 160, 170, 180],
        "Solids": [20000, 21000, 22000, 23000],
        "Chloramines": [5, 6, 7, 8],
        "Conductivity": [400, 410, 420, 430],
        "Organic_carbon": [3.0, 3.2, 3.5, 4.0],
        "Turbidity": [3.0, 3.1, 3.2, 3.3]
    })

    processed_df = preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0  
    assert len(processed_df) <= len(df)  