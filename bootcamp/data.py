from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class Data(ABC):
    """
    Class that represents a generic datasource.
    """

    rename_columns_dict = {
        'country_code': 'Country',
        'cac': 'Client',
        'month_code': 'Date',
        'invoiced_sales': 'InvoiceSales',
        'volume_primary_units': 'VolumeUnits',
        'inventory_cost': 'InventoryCost',
        'bravo_ww_fran_name': 'CompanyGlobalGroup',
        'bravo_fran_name': 'CompanyRegionalGroup',
        'bravo_sub_fran_name': 'CompanyArea',
        'bravo_major_name': 'ProductCategory',
        'bravo_minor_name': 'ProductSubCategory',
        'product_code': 'ProductID'
    }

    columns_to_get_id = [
        'Country',
        'Client',
        'CompanyGlobalGroup',
        'CompanyRegionalGroup',
        'CompanyArea',
        'ProductCategory',
        'ProductSubCategory',
        'ProductID'
    ]

    columns_to_convert_to_numeric = [
        'InvoiceSales',
        'VolumeUnits',
        'InventoryCost'
    ]

    columns_to_convert_to_date = [
        'Date'
    ]

    def __init__(self, data_path):
        self.data_path = data_path

    def _load_raw_data(self):
        """
        Load the data from the defined path to a pandas dataframe
        :param path: str with the file path
        :return: pandas.DataFrame
        """
        logging.info("Loading raw data...")
        df = pd.read_csv(self.data_path, low_memory=False)
        return df

    @staticmethod
    def _rename_columns(df, rename_dict):
        """
        Renames the columns using the defined class rename dictionary.
        :param df: DataFrame with the data to be renamed
        :param rename_dict: Dict with the corresponding renaming names
        :return: DataFrame with the columns renamed
        """
        logging.info("Renaming columns...")
        return df.rename(columns=rename_dict)

    @staticmethod
    def _get_id(df, columns_to_id):
        """
        Gets the id value for each of the rows in each of the columns defined in the class columns list.
        :param df: DataFrame with the data to get the ids
        :param columns_to_id: List with the columns to be evaluated
        :return: DataFrame with the ids
        """
        logging.info("Getting ids...")

        for column in columns_to_id:
            if column in df.columns:
                df[column] = df[column].str.split('_').str[-1]

        return df

    @staticmethod
    def _remove_nan_values(df):
        """
        Removes missing values.
        :param df: DataFrame to be cleaned
        :return: Dataframe without missing values
        """
        logging.info("Removing missing values...")
        return df.dropna()

    @abstractmethod
    def run(self):
        pass


class SalesData(Data):
    """
    Class that gets the sales (stock) data and performs all the required cleaning steps
    """

    @staticmethod
    def _convert_to_numeric(df, columns_to_numeric):
        """
        Converts numeric columns from strings to float and rounds the values to two decimal places.
        :param df: DataFrame with data to be converted
        :param columns_to_numeric: List columns to be converted
        :return: Dataframe with the converted data
        """
        logging.info("Converting values to numeric...")

        for column in columns_to_numeric:
            if column in df.columns:
                df.replace({column: {',': '.'}}, regex=True, inplace=True)
                df[column] = df[column].astype('float').round(2)

        return df

    @staticmethod
    def _convert_to_date(df, columns_to_date):
        """
        Converts the dates columns from string to dates.
        :param df: DataFrame with the data to be converted
        :param columns_to_date: List of columns to be converted
        :return: DataFrame with converted dates
        """
        logging.info("Converting dates to date...")

        for column in columns_to_date:
            if column in df.columns:
                df.replace({column: {'\.0': ''}}, regex=True, inplace=True)
                df[column] = pd.to_datetime(df[column], format='%Y%m')

        return df

    def run(self):
        """
        Performs all the required steps to reach the cleaned data
        :return: DataFrame with cleaned sales data
        """

        # load raw data
        sales_dataset = self._load_raw_data()

        # renames the columns
        sales_dataset = self._rename_columns(sales_dataset, self.rename_columns_dict)

        # get the ids
        sales_dataset = self._get_id(sales_dataset, columns_to_id= self.columns_to_get_id)

        # convert columns to numeric
        sales_dataset = self._convert_to_numeric(sales_dataset, columns_to_numeric = self.columns_to_convert_to_numeric)

        # remove nan values
        sales_dataset = self._remove_nan_values(sales_dataset)

        # convert dates to date
        sales_dataset = self._convert_to_date(sales_dataset, self.columns_to_convert_to_date)

        return sales_dataset


class ProductData(Data):
    """
    Class that gets the product data and performs all the required cleaning steps
    """

    def run(self):
        """
        Performs all the required steps to reach the cleaned data
        :return: DataFrame with cleaned sales data
        """

        # load raw data
        product_dataset = self._load_raw_data()

        # renames the columns
        product_dataset = self._rename_columns(product_dataset, self.rename_columns_dict)

        # get the ids
        product_dataset = self._get_id(product_dataset, columns_to_id=self.columns_to_get_id)

        # remove nan values
        product_dataset = self._remove_nan_values(product_dataset)

        return product_dataset


class FullData(Data):

    columns_to_log_transform = [
        'InvoiceSales',
        'VolumeUnits',
        'InventoryCost',
        'UnitaryInvoiceSales',
        'UnitaryInventoryCost'
    ]

    sale_labels = (
        pd.DataFrame({
            'InvoiceSalesCategories':
                ['>0', '=0', '=0', '<0', '>0', '<0', '>0', '=0', '=0', '<0', '<0', '>0', '<0', '=0', '>0'],
            'VolumeUnitsCategories':
                ['>0', '=0', '>0', '<0', '>0', '=0', '=0', '>0', '<0', '>0', '<0', '<0', '>0', '<0', '<0'],
            'InventoryCostCategories':
                ['>0', '=0', '>0', '<0', '=0', '=0', '=0', '=0', '<0', '>0', '=0', '<0', '=0', '=0', '=0'],
            'SalesLabel':
                ['Normal', 'No info', 'Product giveaway', 'Product and credit return', 'Services', 'Credit return',
                 'Price correction', 'Error', 'Product return', 'Client compensation', 'Error', 'Error', 'Error',
                 'Error', 'Error']
        })
    )

    def __init__(self, stock_dataset, product_dataset, perform_final_cleaning=True):
        self.stock_dataset = stock_dataset
        self.product_dataset = product_dataset
        self.perform_final_cleaning = perform_final_cleaning

    @staticmethod
    def _merge_datasets(stock_dataset, product_dataset):
        """
        Merges the stock and product datasets.
        :param stock_dataset:
        :param product_dataset:
        :return:
        """
        logging.info("Merging both datasets...")

        full_dataset = stock_dataset.merge(product_dataset, on=['ProductID'], how='outer')

        return full_dataset

    @staticmethod
    def _label_sales(df, label_df):
        """
        Labels the sales using the supplied dataframe.
        :param df: DataFrame to be labeled
        :param label_df: DataFrame with the labels to use
        :return: DataFrame with the labeled sales
        """

        logging.info("Labeling sales...")

        # define columns to convert to categories
        columns = ['InvoiceSales', 'VolumeUnits', 'InventoryCost']

        # creates category columns
        for column in columns:
            df[f'{column}Categories'] = '>0'
            df.loc[df[column] < 0, f'{column}Categories'] = '<0'
            df.loc[df[column] == 0, f'{column}Categories'] = '=0'

        # creates the sale labels feature
        df_with_categories = (
            df
            .merge(label_df,
                   on=['InvoiceSalesCategories', 'VolumeUnitsCategories', 'InventoryCostCategories'],
                   how='left')
            .drop(columns=['InvoiceSalesCategories', 'VolumeUnitsCategories', 'InventoryCostCategories'])
        )

        return df_with_categories

    @staticmethod
    def _correct_open_sales_date(df, date_column):
        """
        Corrects the dates of the sales that are still open.
        It is assumed that these dates will be done in the next month.
        :param df: DataFrame with the dates to be corrected
        :return: DataFrame with the corrected dates
        """

        logging.info("Correcting open sales date...")

        next_month = df[date_column].max() + pd.DateOffset(months=1)
        df[date_column].replace('1900-01-01', next_month, inplace=True)

        return df

    @staticmethod
    def _add_year_and_month(df, date_column):
        """
        Converts the specified date column to a month and year columns.
        :param df: DataFrame with the date column
        :param date_column: str with the date column to be converted
        :return: DataFrame without the date column and with two new columns (year and month)
        """

        logging.info("Adding year and month...")

        df_with_month_and_year = (
            df
                .assign(Year=pd.DatetimeIndex(df[date_column]).year)
                .assign(Month=pd.DatetimeIndex(df[date_column]).month)
        )

        return df_with_month_and_year

    @staticmethod
    def _select_sales_type(df, sales_label_column, sales_type):
        """
        Select only the objective sales label from the supplied sales data.
        :param df: DataFrame with the sales data
        :param sales_label_column: str with the column with the sales label column
        :param sales_type: list with the sales labels to be considered
        :return: DataFrame with the defined sales labels
        """

        logging.info("Selecting labels...")

        df = (
            df
            .loc[df[sales_label_column].isin(sales_type)]
        )

        return df

    @staticmethod
    def _add_unitary_features(df):
        """
        Adds the unitary
        :return: DataFrame with unitary features
        """

        logging.info("Adding unitary prices and costs...")

        df = (
            df
            .assign(UnitaryInvoiceSales=lambda x: x['InvoiceSales'] / x['VolumeUnits'])
            .assign(UnitaryInventoryCost=lambda x: x['InventoryCost'] / x['VolumeUnits'])
        )

        return df

    @staticmethod
    def _column_log_transformation(df, columns):
        """
        Renames the columns using the defined class rename dictionary.
        :param df: DataFrame with the data to be renamed
        :param rename_dict: Dict with the corresponding renaming names
        :return: DataFrame with the columns renamed
        """

        logging.info("Adding log transformations...")

        for column in columns:
            df[f'{column}Log'] = np.log(df[column])

        return df

    def run(self):
        """
        Runs all the required steps to reach a the full dataset.
        """

        # merges both datasets
        full_dataset = self._merge_datasets(self.stock_dataset, self.product_dataset)

        # remove null values
        full_dataset = self._remove_nan_values(full_dataset)

        # adds the sale labels
        full_dataset = self._label_sales(full_dataset, self.sale_labels)

        # corrects open sales
        full_dataset = self._correct_open_sales_date(full_dataset, 'Date')

        if self.perform_final_cleaning:

            # selects only the normal type of sales
            full_dataset = self._select_sales_type(full_dataset, 'SalesLabel', ['Normal'])

            # add the year and months features
            full_dataset = self._add_year_and_month(full_dataset, 'Date')

            # add unitary price and cost
            full_dataset = self._add_unitary_features(full_dataset)

            # performs the log transformation of the indicated columns
            full_dataset = self._column_log_transformation(full_dataset, columns = self.columns_to_log_transform)

        return full_dataset


class ModelData:
    """
    Class that prepares the data to
    """

    fold_date_dict = {
        'first_fold': '2019-07-01',
        'second_fold': '2019-06-01',
        'third_fold': '2019-05-01'
    }

    def __init__(self, full_dataset):
        self.full_dataset = full_dataset

    @staticmethod
    def _get_unique_values(df, column):
        """
        Evaluates all the datasets in the dataset list and returns a list with the unique values of the supplied column.
        :param dataset_list: list of datasets with sales data
        :param column: str with the column to be evaluated
        :return: list with the unique values
        """
        logging.info("Getting unique values...")

        unique_values = df[column].unique()

        return unique_values

    @staticmethod
    def _encoder_creation(evaluated_list):
        """
        Creates a dictionary of encoding. From id to encoded and from encoded to id
        :param evaluated_list: list to be used to encode the data
        :return: dict of encodings
        """
        logging.info("Performing encoding...")

        id_to_encoded = {x: i for i, x in enumerate(evaluated_list)}
        encoded_to_id = {i: x for i, x in enumerate(evaluated_list)}

        return id_to_encoded, encoded_to_id

    @staticmethod
    def _apply_encoding(df, column, encoder):
        """
        Applies the encoding to the specified column in the given dataset.
        :param df: DataFrame with the column to be encoded
        :param column: str of the list to be encoded
        :param encoder: encoder to be used
        :return:
        """
        logging.info("Applying encoding...")

        df[column] = df[column].map(encoder)
        return df

    def _encode_dataset(self, df, client2client_encoded, item_family2item_family_encoded):
        """
        Prepares the dataset to be used at the model
        :param df: DataFrame with the data to be prepared
        :return: DataFrame with the relevant data to be used at the model
        """
        logging.info("Encoding dataframe...")

        prepared_df = (
            df
            .pipe(self._apply_encoding, 'Client', client2client_encoded)
            .pipe(self._apply_encoding, 'ProductSubCategory', item_family2item_family_encoded)
        )

        return prepared_df

    @staticmethod
    def _encode_list(list, encoding_dict):
        """
        Performs the encoding of a list
        :param list: List with the data to be encoded
        :param encoding_dict: Dict with the encoding data
        :return: List with the data encoded
        """
        logging.info("Encoding list...")

        list_encoded = np.array([encoding_dict[i] for i in list])

        return list_encoded

    @staticmethod
    def _divide_dataset_by_date(df, date_column, date):
        """
        Divides the dataset into train and test datasets using the supplied date.
        :param df: DataFrame with the data to be divided
        :param date_column: str with the name of the column with the dates
        :param date: str with the limit date to be used
        :return: DataFrame with the train and test datasets
        """

        logging.info("Dividing datasets by date...")

        train_dataset = df.query(f'{date_column} < "{date}"')
        test_dataset = df.query(f'{date_column} == "{date}"')

        return train_dataset, test_dataset

    def _prepare_folds(self, df, fold_date_dict):
        """
        Create the folds defined in the dictionary using the supplied dataset.
        :param df: DataFrame with the data to be used in the fold creation
        :param fold_date_dict:
        :return: Dict[List] with the train and test datasets for each fold
        """

        logging.info("Creating folds...")

        # initializes the fold dictionary
        fold_data_dict = {}

        # creates the folds
        for fold in fold_date_dict.keys():
            train_dataset, test_dataset = self._divide_dataset_by_date(df, 'Date', fold_date_dict[fold])
            fold_data_dict[fold] = [train_dataset, test_dataset]

        return fold_data_dict

    @staticmethod
    def _save_data(object_to_save, save_name, save_folder):
        """
        Saves the object with the defined name in the defined folder
        :param object_to_save: object to be saved
        :param save_name: str to use as saving name
        :param save_folder: str with the path to save
        """

        logging.info("Saving data...")

        pickle.dump(object_to_save, open(f'{save_folder}/{save_name}.pickle', 'wb'))

    def run(self, save_folder):
        """
        Runs all the cleaning and preparation phases
        """

        # get the unique clients
        unique_clients = self._get_unique_values(self.full_dataset, 'Client')

        # gets the unique items
        unique_items_family = self._get_unique_values(self.full_dataset, 'ProductSubCategory')

        # gets the items encoding
        item_family2item_family_encoded, item_family_encoded2item_family = self._encoder_creation(unique_items_family)

        # gets the client encoding
        client2client_encoded, client_encoded2client = self._encoder_creation(unique_clients)

        # encode the dataset
        model_dataset_encoded = self._encode_dataset(self.full_dataset,
                                                     client2client_encoded,
                                                     item_family2item_family_encoded)

        # encode unique clients
        unique_clients_encoded = self._encode_list(unique_clients, client2client_encoded)

        # encode unique items
        unique_items_encoded = self._encode_list(unique_items_family, item_family2item_family_encoded)

        # get fold data
        folds = self._prepare_folds(model_dataset_encoded, self.fold_date_dict)

        # Saves unique clients
        self._save_data(unique_clients_encoded, 'unique_clients_encoded', save_folder)

        # Saves unique items
        self._save_data(unique_items_encoded, 'unique_items_encoded', save_folder)

        # Saves folds dict
        self._save_data(folds, 'folds_dict', save_folder)

        return folds






