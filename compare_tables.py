from typing import Optional, List
import pandas as pd
import numpy as np


def compare_strictly(
    first_df: pd.DataFrame, second_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return None
    diff_mask = (first_df != second_df) & ~(first_df.isnull() & second_df.isnull())
    differences = format_differences(first_df, second_df, diff_mask)
    return differences


def compare_with_tolerence(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return None
    diff_mask = pd.DataFrame()
    non_numeric = first_df.select_dtypes([int, float])
    non_numeric_columns = [c for c in first_df.columns if c not in non_numeric]
    for column in non_numeric:
        diff_mask[column] = ~np.isclose(
            first_df.loc[:, column],
            second_df.loc[:, column],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=True,
        )
    diff_mask[non_numeric_columns] = (
        first_df.loc[:, non_numeric_columns] != second_df.loc[:, non_numeric_columns]
    ) & ~(
        first_df.loc[:, non_numeric_columns].isnull()
        & second_df.loc[:, non_numeric_columns].isnull()
    )
    differences = format_differences(
        first_df, second_df, diff_mask.loc[:, first_df.columns]
    )
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
