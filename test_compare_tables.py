import numpy as np
import pandas as pd
import pytest
from compare_tables import compare_strictly, compare_with_tolerence


def test_compare_strict_identical_tables(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = compare_strictly(simple_dataframe, second_df)

    assert result is None


def test_compare_strict_identical_tables_with_nulls(simple_dataframe):
    simple_dataframe.loc[2, "float_column"] = np.nan
    second_df = simple_dataframe.copy()

    result = compare_strictly(simple_dataframe, second_df)

    assert result is None


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

    result = compare_strictly(simple_dataframe, second_df)
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

    result = compare_strictly(simple_dataframe, second_df)
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

    result = compare_strictly(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_identical_tables(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01
    )

    assert result is None


def test_compare_with_tolerance_identical_tables_with_nulls(simple_dataframe):
    simple_dataframe.loc[2, "float_column"] = np.nan
    second_df = simple_dataframe.copy()

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01, relative_tolerance=1e-6
    )

    assert result is None


def test_compare_with_tolerance_int_difference(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "int_column"] = 19
    second_df.loc[2, "int_column"] = 35
    expected_index = pd.MultiIndex.from_tuples(
        [(1, "int_column")], names=["id", "column"]
    )
    expected = pd.DataFrame(
        {
            "from": [22],
            "to": [19],
        },
        index=expected_index,
        dtype=np.dtype("O"),
    )

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0, relative_tolerance=0.1
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_str_difference(simple_dataframe):
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

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01, relative_tolerance=1e-6
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_float_difference(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.loc[0, "float_column"] = 1.25
    second_df.loc[1, "float_column"] = 2.11
    expected_index = pd.MultiIndex.from_tuples(
        [(0, "float_column")], names=["id", "column"]
    )
    expected = pd.DataFrame(
        {
            "from": [1.1],
            "to": [1.25],
        },
        index=expected_index,
        dtype=np.dtype("O"),
    )

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.1, relative_tolerance=0
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)
