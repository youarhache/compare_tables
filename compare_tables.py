from typing import Optional, List
import pandas as pd
import numpy as np
from tables_structure_checker import check_tables_structure


@check_tables_structure
def compare_strictly(
    first_df: pd.DataFrame, second_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return pd.DataFrame(
            data=np.zeros(first_df.shape),
            columns=first_df.columns,
            index=first_df.index,
            dtype=bool,
        )
    diff_mask = _get_strict_differences_locations(first_df, second_df)
    return diff_mask


@check_tables_structure
def compare_with_tolerence(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> Optional[pd.DataFrame]:
    if first_df.equals(second_df):
        return pd.DataFrame(
            data=np.zeros(first_df.shape),
            columns=first_df.columns,
            index=first_df.index,
            dtype=bool,
        )
    diff_mask = pd.DataFrame()
    non_numeric = first_df.select_dtypes([int, float])
    non_numeric_columns = [c for c in first_df.columns if c not in non_numeric]
    for column in non_numeric:
        # return location where numerical values are not close to each other
        diff_mask[column] = ~np.isclose(
            first_df.loc[:, column],
            second_df.loc[:, column],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=True,
        )
    diff_mask[non_numeric_columns] = _get_strict_differences_locations(
        first_df.loc[:, non_numeric_columns], second_df.loc[:, non_numeric_columns]
    )
    return diff_mask.loc[:, first_df.columns]


def _get_strict_differences_locations(
    first_df: pd.DataFrame, second_df: pd.DataFrame
) -> pd.DataFrame:
    return (first_df != second_df) & ~(first_df.isnull() & second_df.isnull())


def get_formatted_different_values(
    first_df: pd.DataFrame, second_df: pd.DataFrame, diff_mask: pd.DataFrame
) -> pd.DataFrame:
    diff_locations = np.where(diff_mask)
    changed_from = first_df.values[diff_locations]
    changed_to = second_df.values[diff_locations]
    diff_stacked = diff_mask.stack()
    changed_only = diff_stacked.loc[diff_stacked]
    changed_only.index.names = ["id", "column"]
    return pd.DataFrame(
        {"from": changed_from, "to": changed_to}, index=changed_only.index
    )


def get_column_difference_percentages(
    diff_mask: pd.DataFrame, column_types: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "Column name": diff_mask.columns.values,
            "Column type": column_types,
            r"% of difference": 100
            * (diff_mask.sum(axis=0) / diff_mask.shape[0]).values,
        }
    )


def get_number_differenes_per_row(
    diff_mask: pd.DataFrame, limit: int = 5
) -> pd.DataFrame:
    output_column_name = "Number of differences"
    number_of_row_diffs = pd.DataFrame(
        data={output_column_name: diff_mask.sum(axis=1).sort_values(ascending=False)}
    )
    return number_of_row_diffs.nlargest(limit, output_column_name, keep="first")


def get_most_common_different_values_per_column(
    formatted_differences: pd.DataFrame, limit: int = 5
) -> pd.DataFrame:
    differences = formatted_differences.copy().reset_index()
    df_agg = (
        differences.groupby(["column", "from", "to"], as_index=True, dropna=False)
        .agg({"id": "nunique"})
        .rename(columns={"id": "count"})
    )
    return df_agg.nlargest(limit, columns="count", keep="first")
