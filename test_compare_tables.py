from unittest import result
import pytest
import numpy as np
import pandas as pd
from compare_tables import compare_dataframes


def test_compare_strict_identical_tables(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = compare_dataframes(simple_dataframe, second_df)


def test_compare_strict_identical_tables_with_nulls(simple_dataframe):
    simple_dataframe.loc[2, "float_column"] = np.nan
    second_df = simple_dataframe.copy()

    result = compare_dataframes(simple_dataframe, second_df)


def test_compare_strict_int_difference(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.loc[2, "int_column"] = 7
    expected_index = pd.MultiIndex.from_tuples(
        [(2, "int_column")], names=["id", "column"]
    )
    expected = pd.DataFrame(
        {
            "from": [33],
            "to": [7],
        },
        index=expected_index,
        dtype=np.dtype("O"),
    )

    result = compare_dataframes(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_str_difference(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "str_column"] = "ZZ"
    expected_index = pd.MultiIndex.from_tuples(
        [(1, "str_column")], names=["id", "column"]
    )
    expected = pd.DataFrame(
        {
            "from": ["bb"],
            "to": ["ZZ"],
        },
        index=expected_index,
        dtype=np.dtype("O"),
    )

    result = compare_dataframes(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_float_difference(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.loc[0, "float_column"] = 3.14
    expected_index = pd.MultiIndex.from_tuples(
        [(0, "float_column")], names=["id", "column"]
    )
    expected = pd.DataFrame(
        {
            "from": [1.1],
            "to": [3.14],
        },
        index=expected_index,
        dtype=np.dtype("O"),
    )

    result = compare_dataframes(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)
