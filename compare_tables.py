from typing import Optional, List
from matplotlib.font_manager import findSystemFonts
import pandas as pd
import numpy as np


def compare_dataframes(
    first_df: pd.DataFrame, second_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return None
    diff_mask = (first_df != second_df) & ~(first_df.isnull() & second_df.isnull())
    differences = format_differences(first_df, second_df, diff_mask)
    return differences


def compare_with_tolerence(
    first_df: pd.DataFrame, second_df: pd.DataFrame, tolerances: List[float]
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return None
    assert is_values_numeric(first_df) and is_values_numeric(second_df)
    diff_mask = pd.DataFrame()
    for column in first_df.columns:
        diff_mask[column] = first_df.loc[:, column].isclose(second_df.loc[:, column])
    differences = format_differences(first_df, second_df, diff_mask)
    return differences


def format_differences(first_df, second_df, diff_mask):
    diff_locations = np.where(diff_mask)
    changed_from = first_df.values[diff_locations]
    changed_to = second_df.values[diff_locations]
    diff_stacked = diff_mask.stack()
    changed_only = diff_stacked.loc[diff_stacked]
    changed_only.index.names = ["id", "column"]
    return pd.DataFrame(
        {"from": changed_from, "to": changed_to}, index=changed_only.index
    )


def is_same_dtypes(first_df, second_df):
    first_df = first_df.convert_dtypes()
    second_df = second_df.convert_dtypes()
    return first_df.dtypes.equals(second_df.dtypes)
