"""Preprocessing utilities for clustering workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean incoming dataframe before clustering.

    Steps:
    - Drop duplicated rows
    - Replace inf/-inf with NaN
    - Keep numeric columns only
    - Fill missing values using median per column
    """

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.select_dtypes(include=[np.number])

    if cleaned.empty:
        return cleaned

    return cleaned.fillna(cleaned.median(numeric_only=True))


def scale_data(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Scale numerical data using StandardScaler."""

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler
