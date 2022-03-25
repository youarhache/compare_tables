from compare_tables import compare_dataframes
import pytest
import pandas as pd


@pytest.fixture()
def simple_dataframe():
    df = pd.DataFrame(
        {
            "int_column": [11, 22, 33],
            "str_column": ["aa", "bb", "cc"],
            "float_column": [1.1, 2.2, 3.3],
        }
    )

    return df


def test_compare_identical_tables(simple_dataframe):
    second_df = simple_dataframe.copy()
    result = compare_dataframes(simple_dataframe, second_df)

    assert result is None
