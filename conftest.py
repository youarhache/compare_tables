import pytest
import numpy as np
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


@pytest.fixture
def dataframe_of_false_values(simple_dataframe):
    df = pd.DataFrame(
        np.zeros(simple_dataframe.shape),
        columns=simple_dataframe.columns,
        index=simple_dataframe.index,
        dtype=bool,
    )

    return df
