import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(data: pd.DataFrame, test_size: float=0.2, random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(['RainTomorrow'], axis=1)
    y = data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test