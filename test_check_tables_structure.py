import pandas as pd
import numpy as np
from check_tables_structure import ChechTablesStructure


def test_check_tables_types_when_equal(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = ChechTablesStructure.is_same_dtypes(simple_dataframe, second_df)

    assert result is True


def test_check_tables_types_when_infered_types_equal(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df.astype(np.dtype("O"))

    result = ChechTablesStructure.is_same_dtypes(simple_dataframe, second_df)

    assert result is True


def test_check_tables_types_when_infered_types_different(simple_dataframe):
    second_df = pd.DataFrame(
        {
            "int_column": pd.Series([11, 22, 33], dtype=np.dtype("O")),
            "str_column": pd.Series(["aa", "bb", "ccc"], dtype=np.dtype("str")),
            "float_column": pd.Series(["1.1", "2.2", "3.3"], dtype=np.dtype("str")),
        }
    )

    result = ChechTablesStructure.is_same_dtypes(simple_dataframe, second_df)

    assert result is False


def test_check_tables_columns_when_equal(simple_dataframe):
    second_df = simple_dataframe.copy()

    result = ChechTablesStructure.is_same_columns(simple_dataframe, second_df)

    assert result is True


def test_check_tables_columns_when_missing_column_in_first(simple_dataframe):
    second_df = simple_dataframe.copy()
    second_df["new_column"] = np.nan

    result = ChechTablesStructure.is_same_columns(simple_dataframe, second_df)

    assert result is False


def test_check_tables_columns_when_missing_column_in_second(simple_dataframe):
    second_df = simple_dataframe.drop(columns=["float_column"])

    result = ChechTablesStructure.is_same_columns(simple_dataframe, second_df)

    assert result is False
