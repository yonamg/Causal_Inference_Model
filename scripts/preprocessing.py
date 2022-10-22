import sys

import numpy as np
import pandas as pd
import sys
import os

# from logger import Logger


from sklearn.preprocessing import LabelEncoder

# sys.path.append(os.path.abspath(os.path.join("./")))


class PreProcess:
    def __init__(self, df: pd.DataFrame):
        """Initialize the PreProcess class.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
        """
        try:
            self.df = df
            # self.logger = Logger("preprocessing.log").get_app_logger()
            # self.logger.info("Successfully Instantiated Outlier Class Object")
        except Exception:
            # self.logger.exception("Failed to Instantiate Preprocessing Class Object")
            sys.exit(1)

    def convert_to_datetime(self, df, column: str):
        """Convert a column to a datetime.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            column (str): dataframe column to be converted
        """
        df[column] = pd.to_datetime(df[column], errors="coerce")
        # self.logger.info(
        # 'Converted datetime columns to datetime')
        return df

    def convert_to_float(self, df, column: str):
        """Convert column to float.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            column (str): Column to be converted to string
        """
        self.df[column] = df[column].astype(float)
        # self.logger.info("Successfully converted to float columns")
        return self.df

    def drop_variables(self, df):
        """Drop variables based on a percentage.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            percentage(int): Percentage of variables to be dropped
        """
        df_before_filling = df.copy()
        df = df[df.columns[df.isnull().mean() < 0.3]]
        missing_cols = df.columns[df.isnull().mean() > 0]
        print(missing_cols)
        # self.logger.info("Missing columns are: ", missing_cols)
        return df, df_before_filling, missing_cols

    def df_drop_columns(self, df) -> pd.DataFrame:
        """Drop variables based on a percentage.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            percentage(int): Percentage of variables to be dropped
        
        Returns:
            df (pd.DataFrame): A dataframe with dropped columns
        """
        missing_cols = df.columns[df.isnull().mean() > 0]
        # self.logger.info(f"Missing columns are: {missing_cols}")
        df = df.drop(missing_cols, axis=1)
        return df

    def drop_column(self, df, column: str) -> pd.DataFrame:
        """Drop a column.
        Args:
            df (DataFrame): A dataframe to be preprocessed
            column (str): column to be dropped
        Returns:
            df (DataFrame): A dataframe with dropped column
        """
        # self.logger.info(f"Dropping column: {column}")
        df = df.drop(column, axis=1)
        # self.logger.info(f"Dropped column: {column}")
        return df

    def label_encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Label encode the target variable.
        Parameters
        ----------
        df: Pandas Dataframe
            This is the dataframe containing the features and target variable.
        columns: list
        Returns
        -------
        The function returns a dataframe with the target variable encoded.
        """
        # Label Encoding

        label_encoded_columns = []
        # For loop for each columns
        for col in columns:
            # We define new label encoder to each new column
            le = LabelEncoder()
            # Encode our data and create new Dataframe of it,
            # notice that we gave column name in "columns" arguments
            column_dataframe = pd.DataFrame(le.fit_transform(df[col]), columns=[col])
            # and add new DataFrame to "label_encoded_columns" list
            label_encoded_columns.append(column_dataframe)

        # Merge all data frames
        label_encoded_columns = pd.concat(label_encoded_columns, axis=1)
        return label_encoded_columns

    def clean_feature_name(self, df):
        """Clean labels of the dataframe.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
        """
        df.columns = [column.replace(" ", "_").lower() for column in df.columns]
        # self.logger.info("Cleaned feature names")
        return df

    def rename_columns(self, df: pd.DataFrame, column: str, new_column: str):
        """Rename a column.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            column (str): column to be renamed
            new_column (str): New column name
        """
        df[column] = df[column].rename(new_column)
        dfRenamed = df.rename({column: new_column}, axis=1)
        return dfRenamed

    def fill_numerical_variables(self, df):
        """Fill numerical variables.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
        """
        df_single = df
        cols = df_single.columns
        num_cols = df_single.select_dtypes(include=np.number).columns
        df_single.loc[:, num_cols] = df_single.loc[:, num_cols].fillna(
            df_single.loc[:, num_cols].median()
        )
        print(num_cols)
        print(df_single.loc[:, num_cols].median())
        # self.logger.info(
        #     'Filled missing numerical variables')
        return cols, df_single, num_cols

    def fill_categorical_variables(self, df, cols, num_cols, df_single):
        """Fill categorical variables.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
            cols(list): List of columns
            num_cols(list): List of numerical columns
            df_single(pd.DataFrame): Dataframe with filled numerical variables
        """
        cat_cols = list(set(cols) - set(num_cols))
        df_single.loc[:, cat_cols] = df_single.loc[:, cat_cols].fillna(
            df.loc[:, cat_cols].mode().iloc[0]
        )
        df_cols = df_single.columns
        print(cat_cols)
        print(df_single.loc[:, cat_cols].mode().iloc[0])
        # self.logger.info("Filled missing categorical variables with mode")
        return df_cols, df_single, cat_cols

    def drop_duplicates(self, df):
        """Drop duplicates.
        Args:
            df (pd.DataFrame): A dataframe to be preprocessed
        """
        df = df.drop_duplicates()

        return df

    def select_correlated_variables(self, df, threshold: float = 0.8) -> list:
        """Select correlated variables.
        Args:
            df (pd.DataFrame): A dataframe to be preprocessed
            threshold (float): Correlation threshold
        
        Returns:
            list: List of correlated variables
        """

        corr_matrix = df.corr()
        corr_matrix = corr_matrix.abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        df = df.drop(to_drop, axis=1)
        # self.logger.info("Selected correlated variables")
        return df

    def replace_outliers_iqr(self, df, columns):
        """Replace outlier data with IQR."""
        try:
            # self.logger.info('Replacing Outlier Data with IQR')
            for col in columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                cut_off = IQR * 1.5
                lower, upper = Q1 - cut_off, Q3 + cut_off

                df[col] = np.where(df[col] > upper, upper, df[col])
                df[col] = np.where(df[col] < lower, lower, df[col])
            return df
        except Exception:
            # self.logger.exception(
            # #     'Failed to Replace Outlier Data with IQR')
            sys.exit(1)