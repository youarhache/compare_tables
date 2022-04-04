import pandas as pd


class DifferentColumnTypesError(BaseException):
    ...


class DifferentColumnsError(BaseException):
    ...


class DifferentIndexesError(BaseException):
    ...


def is_same_dtypes(first_table: pd.DataFrame, second_table: pd.DataFrame):
    first_table = first_table.convert_dtypes()
    second_table = second_table.convert_dtypes()
    return first_table.dtypes.equals(second_table.dtypes)


def is_same_columns(first_table: pd.DataFrame, second_table: pd.DataFrame):
    return first_table.columns.equals(second_table.columns)


def is_same_index(first_table: pd.DataFrame, second_table: pd.DataFrame):
    return first_table.index.equals(second_table.index)


def run_checks(first_table: pd.DataFrame, second_table: pd.DataFrame) -> bool:
    if not is_same_dtypes(first_table, second_table):
        raise DifferentColumnTypesError(
            f"The tables have different column types: \nfirst table:\n{first_table.dtypes.to_dict()}\n\nsecond table:\n{second_table.dtypes.to_dict()}"
        )

    if not is_same_columns(first_table, second_table):
        raise DifferentColumnsError(
            f"The tables have different columns: \nfirst table:\n{first_table.columns.to_dict()}\n\nsecond table:\n{second_table.columns.to_dict()}"
        )

    if not is_same_index(first_table, second_table):
        raise DifferentIndexesError("The tables have different indexes")

    return True


def check_tables_structure(func):
    def inner(first_table, second_table, **kwargs):
        run_checks(first_table, second_table)
        return func(first_table, second_table, **kwargs)

    return inner
