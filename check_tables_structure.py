class ChechTablesStructure:
    @staticmethod
    def is_same_dtypes(first_df, second_df):
        first_df = first_df.convert_dtypes()
        second_df = second_df.convert_dtypes()
        return first_df.dtypes.equals(second_df.dtypes)
