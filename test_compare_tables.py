import numpy as np
import pandas as pd
from compare_tables import (
    compare_strictly,
    compare_with_tolerence,
    get_column_difference_percentages,
    get_formatted_different_values,
    get_most_common_different_values_per_column,
    get_number_differenes_per_row,
)


def test_compare_strict_identical_tables(simple_dataframe, dataframe_of_false_values):
    second_df = simple_dataframe.copy()
    expected = dataframe_of_false_values.copy()

    result = compare_strictly(simple_dataframe, second_df)

    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_identical_tables_with_nulls(
    simple_dataframe, dataframe_of_false_values
):
    simple_dataframe.loc[2, "float_column"] = np.nan
    second_df = simple_dataframe.copy()
    expected = dataframe_of_false_values.copy()

    result = compare_strictly(simple_dataframe, second_df)

    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_int_difference(simple_dataframe, dataframe_of_false_values):
    second_df = simple_dataframe.copy()
    second_df.loc[2, "int_column"] = 7

    expected = pd.DataFrame(
        np.zeros(simple_dataframe.shape),
        columns=simple_dataframe.columns,
        index=simple_dataframe.index,
        dtype=bool,
    )
    expected.loc[2, "int_column"] = True

    result = compare_strictly(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_str_difference(simple_dataframe, dataframe_of_false_values):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "str_column"] = "ZZ"
    expected = dataframe_of_false_values.copy()
    expected.loc[1, "str_column"] = True

    result = compare_strictly(simple_dataframe, second_df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_strict_float_difference(simple_dataframe, dataframe_of_false_values):
    second_df = simple_dataframe.copy()
    second_df.loc[0, "float_column"] = 3.14
    expected = dataframe_of_false_values.copy()
    expected.loc[0, "float_column"] = True

    result = compare_strictly(simple_dataframe, second_df)

    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_identical_tables(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    expected = dataframe_of_false_values.copy()

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01
    )

    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_identical_tables_with_nulls(
    simple_dataframe, dataframe_of_false_values
):
    simple_dataframe.loc[2, "float_column"] = np.nan
    second_df = simple_dataframe.copy()
    expected = dataframe_of_false_values.copy()

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01, relative_tolerance=1e-6
    )

    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_int_difference(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "int_column"] = 19
    second_df.loc[2, "int_column"] = 35
    expected = dataframe_of_false_values.copy()
    expected.loc[1, "int_column"] = True

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0, relative_tolerance=0.1
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_str_difference(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "str_column"] = "ZZ"
    expected = dataframe_of_false_values.copy()
    expected.loc[1, "str_column"] = True

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.01, relative_tolerance=1e-6
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_compare_with_tolerance_float_difference(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    second_df.loc[0, "float_column"] = 1.25
    second_df.loc[1, "float_column"] = 2.11
    expected = dataframe_of_false_values.copy()
    expected.loc[0, "float_column"] = True

    result = compare_with_tolerence(
        simple_dataframe, second_df, absolute_tolerance=0.1, relative_tolerance=0
    )
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


def test_get_comparison_report_when_no_differences():
    difference_mask = pd.DataFrame(
        np.zeros((4, 3)), columns=["A", "B", "C"], dtype=np.dtype("bool")
    )
    column_types = pd.Series([int, str, float])
    expected = pd.DataFrame(
        data={
            "Column name": ["A", "B", "C"],
            "Column type": column_types.values,
            r"% of difference": np.zeros(3),
        }
    )

    result = get_column_difference_percentages(difference_mask, column_types)

    pd.testing.assert_frame_equal(result, expected)


def test_get_comparison_report_when_a_column_quarter_different():
    difference_mask = pd.DataFrame(
        np.zeros((4, 3)), columns=["A", "B", "C"], dtype=np.dtype("bool")
    )
    difference_mask.iloc[1, 2] = True
    column_types = pd.Series([int, str, float])
    expected = pd.DataFrame(
        data={
            "Column name": ["A", "B", "C"],
            "Column type": column_types.values,
            r"% of difference": [0.0, 0.0, 25.0],
        }
    )

    result = get_column_difference_percentages(difference_mask, column_types)

    pd.testing.assert_frame_equal(result, expected)


def test_get_formatted_differences_identical_tables(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    differences = dataframe_of_false_values.copy()
    expected_index = pd.MultiIndex.from_tuples([], names=["id", "column"])
    expected = pd.DataFrame(data=[], columns=["from", "to"], index=expected_index)

    result = get_formatted_different_values(simple_dataframe, second_df, differences)

    pd.testing.assert_frame_equal(result, expected, check_index_type=False)


def test_get_formatted_differences_str_difference(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "str_column"] = "ZZ"
    differences = dataframe_of_false_values.copy()
    differences.loc[1, "str_column"] = True
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

    result = get_formatted_different_values(simple_dataframe, second_df, differences)
    pd.testing.assert_frame_equal(result, expected)


def test_get_formatted_differences_int_difference(
    simple_dataframe, dataframe_of_false_values
):
    second_df = simple_dataframe.copy()
    second_df.loc[1, "int_column"] = 19
    second_df.loc[2, "int_column"] = 35
    differences = dataframe_of_false_values.copy()
    differences.loc[1, "int_column"] = True
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

    result = get_formatted_different_values(simple_dataframe, second_df, differences)
    pd.testing.assert_frame_equal(result, expected)


def test_get_number_differenes_per_row_when_no_differences(
    simple_dataframe, dataframe_of_false_values
):
    differences = dataframe_of_false_values.copy()
    limit = 3
    expected = pd.DataFrame(
        data={
            "Number of differences": [0, 0, 0],
        },
        index=differences.index[:limit],
    )

    result = get_number_differenes_per_row(diff_mask=differences, limit=limit)
    pd.testing.assert_frame_equal(result, expected)


def test_get_number_differenes_per_row_when_differences(
    simple_dataframe, dataframe_of_false_values
):
    differences = dataframe_of_false_values.copy()
    differences.iloc[0, :] = True  # 3 differences in the first line
    differences.iloc[1, :2] = True  # 2 differences in the seconde line
    differences.iloc[2, :2] = True  # 2 differences in the third line
    limit = 2
    expected = pd.DataFrame(
        data={
            "Number of differences": [3, 2],
        },
        index=differences.index[:limit],
    )

    result = get_number_differenes_per_row(diff_mask=differences, limit=limit)
    pd.testing.assert_frame_equal(result, expected)


def test_get_most_common_different_values_per_column():
    limit = 3
    input_index = pd.MultiIndex.from_tuples(
        [
            (1, "str_column"),
            (1, "int_column"),
            (2, "str_column"),
            (3, "int_column"),
            (3, "float_column"),
        ],
        names=["id", "column"],
    )
    expected_index = pd.MultiIndex.from_tuples(
        [("str_column", "aaa", ""), ("float_column", 3.14, 3.2), ("int_column", 6, 7)],
        names=["column", "from", "to"],
    )
    formatted_diff = pd.DataFrame(
        {
            "from": ["aaa", 22, "aaa", 6, 3.14],
            "to": ["", 19, "", 7, 3.2],
        },
        index=input_index,
        dtype=np.dtype("O"),
    )
    expected = pd.DataFrame(data={"count": [2, 1, 1]}, index=expected_index)

    result = get_most_common_different_values_per_column(
        formatted_differences=formatted_diff, limit=limit
    )

    pd.testing.assert_frame_equal(result, expected)
