import numpy as np
import pandas as pd

from bootcamp.models import Models

class Metrics:
    """
    Class that calculates all the required metrics
    """

    def __init__(self, train, test, unique_items):
        self.train = train
        self.test = test
        self.unique_items = unique_items

    @staticmethod
    def _get_unique_clients(df):
        """
        Gets the unique clients of a dataframe.
        :param df: DataFrame to be used in determining the clients
        :return: List unique list of clients.
        """
        unique_clients = df['Client'].unique()
        return unique_clients

    @staticmethod
    def _get_client_previous_purchased_items(train, client):
        """
        Get the purchases items of a specific client.
        :param client: str with the client id
        :return: List of purchase items
        """

        previous_purchased_items = (
            train
            .query(f'Client == {client}')
            ['ProductSubCategory']
            .unique()
        )

        return previous_purchased_items

    @staticmethod
    def _get_client_relevant_items(test, client):
        """
        Get the relevant items for a specific client.
        :param client: str with the client id
        :return: List purchase items
        """

        relevant_items = (
            test
            .query(f'Client == {client}')
            ['ProductSubCategory']
            .unique()
        )

        return relevant_items

    @staticmethod
    def _get_recommended_items(item_ranking, previous_purchased_items, k):
        """
        Uses the item rankings and previous purchase items to build a recommendation list with k items.
        :param item_ranking: DataFrame with the product ranking
        :param previous_purchased_items: List with the previous purchased items
        :param k: int with the number of items to be recommended
        :return: List of recommended items
        """

        recommended_items = (
            item_ranking
            .loc[~item_ranking['ProductSubCategory'].isin(previous_purchased_items)]
            .head(k)
            ['ProductSubCategory']
            .values
        )

        return recommended_items

    @staticmethod
    def _get_recommended_and_relevant_items(recommended_items, relevant_items):
        """
        Uses the recommended and relevant list to determine the relevant items that are recommended.
        :param recommended_items: List with the recommended items
        :param relevant_items: List with the relevant items
        :return: List with the recommended and relevant items
        """

        recommended_and_relevant_items = (
            recommended_items[np.isin(recommended_items, relevant_items)]
        )

        return recommended_and_relevant_items

    @staticmethod
    def _get_client_precision_at_k(recommended_and_relevant_items, k):
        """
        Calculates the precision at k for a specific client.
        :param recommended_and_relevant_items: list with the relevant items that are recommended
        :param k: int with the number of items to be recommended
        :return: float with the value of precision at k
        """
        precision_at_k = len(recommended_and_relevant_items) / k
        return precision_at_k

    @staticmethod
    def _get_client_recall_at_k(recommended_and_relevant_items, relevant_items):
        """
        Calculates the recall at k for a specific client.
        :param recommended_and_relevant_items: list with the relevant items that are recommended
        :param k: int with the number of items to be recommended
        :return: float with the value of precision at k
        """

        if len(relevant_items) != 0:
            recall_at_k = len(recommended_and_relevant_items) / len(relevant_items)
        else:
            recall_at_k = 1

        return recall_at_k

    def _calculate_global_metrics(self, global_recommended_items, precisions_at_k, recalls_at_k, k):
        """
        Calculates the global metrics using the client metrics lists
        :param global_recommended_items: List with all the recommended items
        :param precisions_at_k: List with all the precision at k values
        :param recalls_at_k: List with all the recalls at k
        :return: Dict with all the metrics
        """

        # calculates the global coverage and the median precision and recall at k
        coverage = round(len(np.unique(np.concatenate(global_recommended_items, axis=None))) /
                         len(self.unique_items) * 100, 3)
        mean_precision_at_k = round(np.mean(precisions_at_k) * 100, 3)
        mean_recall_at_k = round(np.mean(recalls_at_k) * 100, 3)

        # creates a dictionary with the results
        results = {
            'k': k,
            'coverage': coverage,
            'mean_precision_at_k': mean_precision_at_k,
            'mean_recall_at_k': mean_recall_at_k
        }

        return results

    def last_month_model_metrics(self, k):
        """
        Calculates the performance metrics for the previous month top sales model.
        :param k: int with the number of items to be suggested
        :return: Dict with the main performance metrics
        """

        # define result lists
        global_recommended_items = []
        precisions_at_k = []
        recalls_at_k = []

        # get unique clients
        unique_clients = self._get_unique_clients(self.test)

        # get the item ranking
        item_ranking = Models().last_month_model_item_ranking(self.train)

        # cycle all the clients
        for client in unique_clients:

            # get previous purchase items
            previous_purchase_items = self._get_client_previous_purchased_items(self.train, client)

            # get relevant purchase items
            relevant_items = self._get_client_relevant_items(self.test, client)

            # get recommended items
            recommended_items = self._get_recommended_items(item_ranking, previous_purchase_items, k)

            # get recommended and relevant items
            recommended_and_relevant_items = self._get_recommended_and_relevant_items(recommended_items, relevant_items)

            # get metrics
            precision_at_k = self._get_client_precision_at_k(recommended_and_relevant_items, k)
            recall_at_k = self._get_client_recall_at_k(recommended_and_relevant_items, relevant_items)

            # adds the precision at k, recall at k and coverage to the specific list
            global_recommended_items.append(recommended_items)
            precisions_at_k.append(precision_at_k)
            recalls_at_k.append(recall_at_k)

            # get the global metric results
            metrics = self._calculate_global_metrics(global_recommended_items, precisions_at_k, recalls_at_k, k)

        return metrics

    def gmf_model_metrics(self, k, model):
        """
        Calculates the performance metrics for the gmf model
        :param k: int with the number of items to be suggested
        :param model: model to be used
        :return: Dict with the main performance metrics
        """

        # define result lists
        global_recommended_items = []
        precisions_at_k = []
        recalls_at_k = []

        # get unique clients
        unique_clients = self._get_unique_clients(self.test)

        # cycle all the clients
        for client in unique_clients:

            # get previous purchase items
            previous_purchase_items = self._get_client_previous_purchased_items(self.train, client)

            # get relevant purchase items
            relevant_items = self._get_client_relevant_items(self.test, client)

            # get the item ranking
            item_ranking = Models().gmf_item_ranking(self.unique_items, client, model)

            # get recommended items
            recommended_items = self._get_recommended_items(item_ranking, previous_purchase_items, k)

            # get recommended and relevant items
            recommended_and_relevant_items = self._get_recommended_and_relevant_items(recommended_items, relevant_items)

            # get metrics
            precision_at_k = self._get_client_precision_at_k(recommended_and_relevant_items, k)
            recall_at_k = self._get_client_recall_at_k(recommended_and_relevant_items, relevant_items)

            # adds the precision at k, recall at k and coverage to the specific list
            global_recommended_items.append(recommended_items)
            precisions_at_k.append(precision_at_k)
            recalls_at_k.append(recall_at_k)

            # get the global metric results
            metrics = self._calculate_global_metrics(global_recommended_items, precisions_at_k, recalls_at_k, k)

        return metrics

    def mlp_model_metrics(self, k, model):
        """
        Calculates the performance metrics for the mlp model
        :param k: int with the number of items to be suggested
        :param model: model to be used
        :return: Dict with the main performance metrics
        """

        # define result lists
        global_recommended_items = []
        precisions_at_k = []
        recalls_at_k = []

        # get unique clients
        unique_clients = self._get_unique_clients(self.test)

        # get client and item characteristics
        client_characteristics_encoding = Models()._create_client_characteristics_encoding(self.train)
        item_characteristics_encoding = Models()._create_item_characteristics_encoding(self.train)

        # cycle all the clients
        for client in unique_clients:

            # get previous purchase items
            previous_purchase_items = self._get_client_previous_purchased_items(self.train, client)

            # get relevant purchase items
            relevant_items = self._get_client_relevant_items(self.test, client)

            # get the item ranking
            item_ranking = Models().mlp_item_ranking(self.test, self.unique_items, client, model,
                                                     client_characteristics_encoding, item_characteristics_encoding)

            # get recommended items
            recommended_items = self._get_recommended_items(item_ranking, previous_purchase_items, k)

            # get recommended and relevant items
            recommended_and_relevant_items = self._get_recommended_and_relevant_items(recommended_items, relevant_items)

            # get metrics
            precision_at_k = self._get_client_precision_at_k(recommended_and_relevant_items, k)
            recall_at_k = self._get_client_recall_at_k(recommended_and_relevant_items, relevant_items)

            # adds the precision at k, recall at k and coverage to the specific list
            global_recommended_items.append(recommended_items)
            precisions_at_k.append(precision_at_k)
            recalls_at_k.append(recall_at_k)

            # get the global metric results
            metrics = self._calculate_global_metrics(global_recommended_items, precisions_at_k, recalls_at_k, k)

        return metrics

    def gmf_mlp_model_metrics(self, k, model):
        """
        Calculates the performance metrics for the gmf mlp model
        :param k: int with the number of items to be suggested
        :param model: model to be used
        :return: Dict with the main performance metrics
        """

        # define result lists
        global_recommended_items = []
        precisions_at_k = []
        recalls_at_k = []

        # get unique clients
        unique_clients = self._get_unique_clients(self.test)

        # get client and item characteristics
        client_characteristics_encoding = Models()._create_client_characteristics_encoding(self.train)
        item_characteristics_encoding = Models()._create_item_characteristics_encoding(self.train)

        # cycle all the clients
        for client in unique_clients:

            # get previous purchase items
            previous_purchase_items = self._get_client_previous_purchased_items(self.train, client)

            # get relevant purchase items
            relevant_items = self._get_client_relevant_items(self.test, client)

            # get the item ranking
            item_ranking = Models().gmf_mlp_item_ranking(self.test, self.unique_items, client, model,
                                                         client_characteristics_encoding, item_characteristics_encoding)

            # get recommended items
            recommended_items = self._get_recommended_items(item_ranking, previous_purchase_items, k)

            # get recommended and relevant items
            recommended_and_relevant_items = self._get_recommended_and_relevant_items(recommended_items, relevant_items)

            # get metrics
            precision_at_k = self._get_client_precision_at_k(recommended_and_relevant_items, k)
            recall_at_k = self._get_client_recall_at_k(recommended_and_relevant_items, relevant_items)

            # adds the precision at k, recall at k and coverage to the specific list
            global_recommended_items.append(recommended_items)
            precisions_at_k.append(precision_at_k)
            recalls_at_k.append(recall_at_k)

            # get the global metric results
            metrics = self._calculate_global_metrics(global_recommended_items, precisions_at_k, recalls_at_k, k)

        return metrics