import re
import pandas as pd


class TablesStructureChecker:
    @staticmethod
    def is_same_dtypes(first_df: pd.DataFrame, second_df: pd.DataFrame):
        first_df = first_df.convert_dtypes()
        second_df = second_df.convert_dtypes()
        return first_df.dtypes.equals(second_df.dtypes)

    @staticmethod
    def is_same_columns(first_df: pd.DataFrame, second_df: pd.DataFrame):
        return first_df.columns.equals(second_df.columns)

    @staticmethod
    def is_same_index(first_df: pd.DataFrame, second_df: pd.DataFrame):
        return first_df.index.equals(second_df.index)
