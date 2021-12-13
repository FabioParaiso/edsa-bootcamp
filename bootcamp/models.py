import numpy as np
import pandas as pd

from keras.layers import Input, Embedding, Flatten, Dense, Dot, Concatenate
from keras.models import Model
import keras


class Models:

    def __init__(self):
        pass

    @staticmethod
    def _select_last_month_sales(df):
        """
        Selects the sales from the last month of the supplied dataset.
        :param df: DataFrame with the data to be evaluated
        :return: DataFrame with the sales from the last month
        """

        # determines the last month date
        last_month = df['Date'].max()

        # gets the last month sales
        last_month_sales = (
            df
            .query(f'Date == "{last_month}"')

        )

        return last_month_sales

    @staticmethod
    def _create_negative_cases_gmf(train):
        """
        Creates negative cases since all the cases we have are positive. It is assumed that all the purchases that were not
        made are possible negative cases.
        :param df: DataFrame with the positive cases
        :return: DataFrame with the positive and negative cases
        """

        # gets the positive cases and assigns a label of 1
        positive_samples = (
            train
            [['Client', 'ProductSubCategory']]
            .drop_duplicates()
            .assign(Label=1)
        )

        # gets the unique combinations of client and item that will be used to evaluate if the created negative cases
        # already exist
        unique_client_item_combinations = (
            train
            .groupby(['Client', 'ProductSubCategory'])
            .agg(Label=('Client', np.size))
            .reset_index()
        )

        # gets the unique client an items
        unique_clients = train['Client'].unique()
        unique_items = train['ProductSubCategory'].unique()

        # creates a random list of clients and items with the same length of the positive cases
        np.random.seed(42)
        negative_clients = np.random.choice(unique_clients, len(positive_samples))
        negative_items = np.random.choice(unique_items, len(positive_samples))

        # creates the negative cases
        negative_samples = (
            pd.DataFrame({
                "Client": negative_clients,
                'ProductSubCategory': negative_items
            })
            .merge(unique_client_item_combinations, on=['Client', 'ProductSubCategory'], how='left')
            .query('Label.isna()')
            .fillna(0)
            .drop_duplicates()
        )

        # concatenates both the positive and negative cases and shuffles the data
        train_dataset = (
            pd.concat([positive_samples, negative_samples])
            .sample(frac=1, random_state=42)
        )

        return train_dataset

    def last_month_model_item_ranking(self, last_month_purchases):
        """
        Calculates a ranking of the previous month most sold products.
        :param last_month_purchases: DataFrame with the last month month purchases
        :return: DataFrame with the ranking of most sold products
        """

        last_month_item_ranking = (
            last_month_purchases
            .pipe(self._select_last_month_sales)
            .groupby('ProductSubCategory')
            .agg(Prediction=('ProductSubCategory', np.size))
            .reset_index()
            .sort_values(by='Prediction', ascending=False)
        )

        return last_month_item_ranking

    @staticmethod
    def _create_negative_cases_mlp(train):
        """
        Creates negative cases since all the cases we have are positive. It is assumed that all the purchases that were not
        made are possible negative cases.
        :param df: DataFrame with the positive cases
        :return: DataFrame with the positive and negative cases
        """

        # gets the positive cases and assigns a label of 1
        positive_samples = (
            train
            [['Client', 'ProductSubCategory', 'Month']]
            .assign(Label=1)
            .drop_duplicates()
        )

        # gets the unique combinations of client and item that will be used to evaluate if the created negative cases
        # already exist
        unique_client_item_combinations = (
            train
            .groupby(['Client', 'ProductSubCategory', 'Month'])
            .agg(Label=('Client', np.size))
            .reset_index()
        )

        # gets the unique client an items
        unique_clients = train['Client'].unique()
        unique_items = train['ProductSubCategory'].unique()
        unique_months = train['Month'].unique()

        # creates a random list of clients and items with the same length of the positive cases
        np.random.seed(42)
        negative_clients = np.random.choice(unique_clients, len(positive_samples))
        negative_items = np.random.choice(unique_items, len(positive_samples))
        negative_month = np.random.choice(unique_months, len(positive_samples))

        # creates the negative cases
        negative_samples = (
            pd.DataFrame({
                "Client": negative_clients,
                'ProductSubCategory': negative_items,
                'Month': negative_month})
            .merge(unique_client_item_combinations, on=['Client', 'ProductSubCategory', 'Month'], how='left')
            .query('Label.isna()')
            .fillna(0)
            .drop_duplicates()
        )

        # concatenates both the positive and negative cases and shuffles the data
        positive_and_negative_samples = (
            pd.concat([positive_samples, negative_samples])
            .sample(frac=1, random_state=42)
        )

        return positive_and_negative_samples

    @staticmethod
    def _log_transform(df, column, n):
        """
        Applies the log transformation to the columns of the supplied dataframe
        :param df: DataFrame to apply the transformation
        :param column: str where to apply the transformation
        :param n: int with the value to sum to the base number
        :return: DataFrame with a column transformed using a log transformation
        """
        df[column] = np.log(df[column] + n)
        return df

    def _create_client_characteristics_encoding(self, train):
        """
        Creates a dataframe with the client characteristics
        :param train: DataFrame with the data to be used
        :return: DataFrame with the client characteristics
        """

        client_characteristics_encoding = (
            train
            .groupby(['Client'])
            .agg(InvoiceSalesLogMean=('InvoiceSalesLog', np.mean),
                 InvoiceSalesLogStd=('InvoiceSalesLog', np.std),
                 VolumeUnitsLogMean=('VolumeUnitsLog', np.mean),
                 VolumeUnitsLogStd=('VolumeUnitsLog', np.std),
                 InventoryCostLogMean=('InventoryCostLog', np.mean),
                 InventoryCostLogStd=('InventoryCostLog', np.std),
                 NumberSales=('Client', np.size),
                 Country=('Country', np.max))
            .pipe(self._log_transform, 'NumberSales', 1)
            .add_prefix('client_')
            .fillna(0)
            .reset_index()
        )

        return client_characteristics_encoding

    def _create_item_characteristics_encoding(self, train):
        """
        Creates a dataframe with the item characteristics
        :param train: DataFrame with the data to be used
        :return: DataFrame with the item characteristics
        """

        item_characteristics_encoding = (
            train
            .groupby(['ProductSubCategory'])
            .agg(InvoiceSalesLogMean=('InvoiceSalesLog', np.mean),
                 InvoiceSalesLogStd=('InvoiceSalesLog', np.std),
                 VolumeUnitsLogMean=('VolumeUnitsLog', np.mean),
                 VolumeUnitsLogStd=('VolumeUnitsLog', np.std),
                 InventoryCostLogMean=('InventoryCostLog', np.mean),
                 InventoryCostLogStd=('InventoryCostLog', np.std),
                 NumberSales=('ProductSubCategory', np.size))
            .pipe(self._log_transform, 'NumberSales', 1)
            .add_prefix('item_')
            .fillna(0)
            .reset_index()
        )

        return item_characteristics_encoding

    def train_gmf_model(self, train, embedding_size, batch_size, epochs, num_clients, num_items):
        """
        Trains a gmf model using the specified settings and respective dataset
        :param train: DataFrame with the input data
        :param embedding_size: int with the embedding layer size
        :param batch_size: int with the batch size to be used
        :param epochs: int with the number of epochs
        :param num_clients: int with the number of clients
        :param num_items: int with the number of items
        :return: trained model
        """

        # add negative samples to the train data
        train = self._create_negative_cases_gmf(train)

        # input layer
        clients_input = Input(shape=(1,), name='clients')
        product_input = Input(shape=(1,), name='products')

        # embedding layers
        client_embedding = Embedding(num_clients,
                                     embedding_size,
                                     embeddings_initializer="he_normal",
                                     embeddings_regularizer=keras.regularizers.l2(1e-6),
                                     name='clients_embedding')(clients_input)
        client_bias = Embedding(num_clients, 1, name='client_bias')(clients_input)
        product_embedding = Embedding(num_items,
                                      embedding_size,
                                      embeddings_initializer="he_normal",
                                      embeddings_regularizer=keras.regularizers.l2(1e-6),
                                      name='products_embedding')(product_input)
        product_bias = Embedding(num_clients, 1, name='item_bias')(product_input)

        # dot product of bot embedding layers at the second axis
        dot = Dot(2, name='dot_product')([client_embedding, product_embedding])

        dot = Flatten(name='dot_flatten')(dot)
        client_bias = Flatten(name='clients_flatten')(client_bias)
        product_bias = Flatten(name='item_flatten')(product_bias)

        # flattening of the dot layer
        sum_layer = dot + client_bias + product_bias

        # sigmoid to get the
        sigmoid = Dense(1, activation='sigmoid', name='classification')(sum_layer)

        # model definition
        model = Model(inputs=[clients_input, product_input], outputs=sigmoid)

        # model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # model fitting
        model.fit([train['Client'].values, train['ProductSubCategory'].values],
                  train['Label'].values,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1)

        return model

    @staticmethod
    def gmf_item_ranking(unique_items, client, model):
        """
        Calculates the item ranking for the gmf models.
        :param unique_items: List with the unique items
        :param client: str with the client identifier
        :param model: model to be used in the predictions
        :return: DataFrame with the rank of the items
        """

        # defines the non purchased items for the client and predicts their probability
        client_array = np.full((len(unique_items),), client)
        prediction_array = model.predict([client_array, unique_items]).reshape(len(unique_items))

        # determines the item ranking
        gmf_item_ranking = (
            pd.DataFrame({
                "ProductSubCategory": unique_items,
                "Prediction": prediction_array
            })
            .sort_values(by='Prediction', ascending=False)
        )

        return gmf_item_ranking

    def train_mlp_model(self, train, batch_size, epochs):
        """
        Trains a mlp model using the specified settings and respective dataset
        :param train: DataFrame with the input data
        :param batch_size: int with the batch size to be used
        :param epochs: int with the number of epochs
        :return: trained model
        """

        client_characteristics_encoding = self._create_client_characteristics_encoding(train)
        item_characteristics_encoding = self._create_item_characteristics_encoding(train)

        # selects the last month of sales for all the training datasets
        train = self._create_negative_cases_mlp(train)

        # selects the last month of sales for all the training datasets
        train = (
            train
            .merge(client_characteristics_encoding, on='Client', how='left')
            .merge(item_characteristics_encoding, on='ProductSubCategory', how='left')
            .drop(columns=['Client', 'ProductSubCategory'])
            .astype(float)
        )

        sales_input = Input(shape=(16,), name='sales_inputs')

        dense_128 = Dense(128, name='dense_128')(sales_input)

        dense_64 = Dense(64, name='dense_64')(dense_128)

        dense_32 = Dense(32, name='dense_32')(dense_64)

        dense_16 = Dense(16, name='dense_16')(dense_32)

        # sigmoid to get the
        sigmoid = Dense(1, activation='sigmoid', name='classification')(dense_16)

        # model definition
        model = Model(inputs=sales_input, outputs=sigmoid)

        # model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # model fitting
        model.fit(train.drop(columns=['Label']).values,
                  train['Label'].values,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1)

        return model

    def mlp_item_ranking(self, test, unique_items, client,
                         model, client_characteristics_encoding, item_characteristics_encoding):
        """
        Calculates the item ranking for the mlp models.
        :param test: DataFrame with the train data
        :param unique_items: List with the unique items
        :param client: str with the client identifier
        :param model: model to be used in the predictions
        :param client_characteristics_encoding: DataFrame with the client encoding
        :param item_characteristics_encoding: DataFrame with the client encoding
        :return: DataFrame with the rank of the items
        """

        # defines the non purchased items for the client and predicts their probability
        client_array = np.full((len(unique_items),), client)
        month_array = np.full((len(unique_items),), test.Month.unique()[0])

        # determines the items to be recommended
        dataset_for_predictions = (
            pd.DataFrame({
                "Client": client_array,
                "ProductSubCategory": unique_items,
                "Month": month_array
            })
            .merge(client_characteristics_encoding, on='Client', how='left')
            .merge(item_characteristics_encoding, on='ProductSubCategory', how='left')
            .drop(columns=['Client', 'ProductSubCategory'])
            .fillna(0)
            .astype(float)
        )

        prediction_array = (
            model
            .predict(dataset_for_predictions.values)
            .reshape(len(unique_items))
        )

        mlp_item_ranking = (
            pd.DataFrame({
                "ProductSubCategory": unique_items,
                "Prediction": prediction_array
            })
            .sort_values(by='Prediction', ascending=False)
        )

        return mlp_item_ranking

    def train_gmf_mlp_model(self, train, batch_size, epochs, embedding_size, num_clients, num_items):
        """
        Trains a gmf model using the specified settings and respective dataset
        :param train_dataset: DataFrame with the input data
        :param batch_size: int with the batch size to be used
        :param epochs: int with the number of epochs
        :return: trained model
        """

        client_characteristics_encoding = self._create_client_characteristics_encoding(train)
        item_characteristics_encoding = self._create_item_characteristics_encoding(train)

        # selects the last month of sales for all the training datasets
        train = self._create_negative_cases_mlp(train)

        # selects the last month of sales for all the training datasets
        train = (
            train
            .merge(client_characteristics_encoding, on='Client', how='left')
            .merge(item_characteristics_encoding, on='ProductSubCategory', how='left')
            .astype(float)
        )

        sales_input = Input(shape=(16,), name='sales_inputs')
        clients_input = Input(shape=(1,), name='clients')
        product_input = Input(shape=(1,), name='products')

        client_embedding = Embedding(num_clients,
                                     embedding_size,
                                     embeddings_initializer="he_normal",
                                     embeddings_regularizer=keras.regularizers.l2(1e-6),
                                     name='clients_embedding')(clients_input)
        client_bias = Embedding(num_clients, 1, name='client_bias')(clients_input)
        product_embedding = Embedding(num_items,
                                      embedding_size,
                                      embeddings_initializer="he_normal",
                                      embeddings_regularizer=keras.regularizers.l2(1e-6),
                                      name='products_embedding')(product_input)
        product_bias = Embedding(num_clients, 1, name='item_bias')(product_input)

        # dot product of bot embedding layers at the second axis
        dot = Dot(2, name='dot_product')([client_embedding, product_embedding])

        dot = Flatten(name='dot_flatten')(dot)
        client_bias = Flatten(name='clients_flatten')(client_bias)
        product_bias = Flatten(name='item_flatten')(product_bias)

        # flattening of the dot layer
        sum_layer = dot + client_bias + product_bias

        dense_128 = Dense(128, name='dense_128')(sales_input)

        dense_64 = Dense(64, name='dense_64')(dense_128)

        dense_32 = Dense(32, name='dense_32')(dense_64)

        dense_16 = Dense(16, name='dense_16')(dense_32)

        gmf_mlp_concatenation = Concatenate()([sum_layer, dense_16])

        dense_16_v2 = Dense(16, name='dense_16_v2')(gmf_mlp_concatenation)

        # sigmoid to get the
        sigmoid = Dense(1, activation='sigmoid', name='classification')(dense_16_v2)

        # model definition
        model = Model(inputs=[sales_input, clients_input, product_input], outputs=sigmoid)

        # model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # model fitting
        model.fit([train.drop(columns=['Label', 'Client', 'ProductSubCategory']).values,
                   train['Client'].values, train['ProductSubCategory'].values],
                  train['Label'].values,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1)

        return model

    def gmf_mlp_item_ranking(self, test, unique_items, client,
                             model, client_characteristics_encoding, item_characteristics_encoding):
        """
        Calculates the item ranking for the mlp models.
        :param test: DataFrame with the train data
        :param unique_items: List with the unique items
        :param client: str with the client identifier
        :param model: model to be used in the predictions
        :param client_characteristics_encoding: DataFrame with the client encoding
        :param item_characteristics_encoding: DataFrame with the client encoding
        :return: DataFrame with the rank of the items
        """

        # defines the non purchased items for the client and predicts their probability
        client_array = np.full((len(unique_items),), client)
        month_array = np.full((len(unique_items),), test.Month.unique()[0])

        # determines the items to be recommended
        dataset_for_predictions = (
            pd.DataFrame({
                "Client": client_array,
                "ProductSubCategory": unique_items,
                "Month": month_array
            })
            .merge(client_characteristics_encoding, on='Client', how='left')
            .merge(item_characteristics_encoding, on='ProductSubCategory', how='left')
            .fillna(0)
            .astype(float)
        )

        prediction_array = (
            model
            .predict([dataset_for_predictions.drop(columns=['Client', 'ProductSubCategory']).values,
                      dataset_for_predictions['Client'].values,
                      dataset_for_predictions['ProductSubCategory'].values])
            .reshape(len(unique_items))
        )

        mlp_item_ranking = (
            pd.DataFrame({
                "ProductSubCategory": unique_items,
                "Prediction": prediction_array
            })
            .sort_values(by='Prediction', ascending=False)
        )

        return mlp_item_ranking
