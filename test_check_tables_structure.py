import pandas as pd
import numpy as np
from tables_structure_checker import is_same_dtypes, is_same_columns, is_same_index


def test_check_tables_types_when_equal(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = is_same_dtypes(simple_dataframe, second_df)

    assert result is True


def test_check_tables_types_when_infered_types_equal(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.astype(np.dtype("O"))

    result = is_same_dtypes(simple_dataframe, second_df)

    assert result is True


def test_check_tables_types_when_infered_types_different(simple_dataframe):
    second_df = pd.DataFrame(
        {
            "int_column": pd.Series([11, 22, 33], dtype=np.dtype("O")),
            "str_column": pd.Series(["aa", "bb", "ccc"], dtype=np.dtype("str")),
            "float_column": pd.Series(["1.1", "2.2", "3.3"], dtype=np.dtype("str")),
        }
    )

    result = is_same_dtypes(simple_dataframe, second_df)

    assert result is False


def test_check_tables_columns_when_equal(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = is_same_columns(simple_dataframe, second_df)

    assert result is True


def test_check_tables_columns_when_missing_column_in_first(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df["new_column"] = np.nan

    result = is_same_columns(simple_dataframe, second_df)

    assert result is False


def test_check_tables_columns_when_missing_column_in_second(simple_dataframe):
    second_df = simple_dataframe.drop(columns=["float_column"])

    result = is_same_columns(simple_dataframe, second_df)

    assert result is False


def test_check_tables_indexes_when_same(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = is_same_index(simple_dataframe, second_df)

    assert result is True


def test_check_tables_indexes_when_different_values(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.set_index(pd.Index([5, 6, 7]), inplace=True)

    result = is_same_index(simple_dataframe, second_df)

    assert result is False


def test_check_tables_indexes_when_different_types(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.set_index(pd.Index(["1", "2", "3"]), inplace=True)

    result = is_same_index(simple_dataframe, second_df)

    assert result is False
