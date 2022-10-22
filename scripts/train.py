"""Causality Scripts."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from causalnex.plots import EDGE_STYLE, NODE_STYLE, plot_structure
from causalnex.discretiser import Discretiser
from IPython.display import Image
# from logger import Logger


class Causal:
    def __init__(self):
        """Initialize the Causality class.
        Args:
            df (pd.DataFrame): dataframe to be preprocessed
        """
        # try:
        #     self.logger = Logger("causal.log").get_app_logger()
        #     self.logger.info("Successfully Instantiated Causal Class Object")
        # except Exception:
        #     self.logger.exception("Failed to Instantiate Causal Class Object")
        #     sys.exit(1)

    def jaccard_similarity(self, y_true: list, y_pred: list) -> float:
        """Calculate the Jaccard similarity.
        Args:
            y_true (list): 1d array-like sparse matrix Ground truth (correct) labels.
            y_pred (list): 1d array-like sparse matrix Predicted labels, as returned by a classifier.
        Returns:
            float: jaccard similarity
        """
        # self.logger.info("Calculating Jaccard Similarity")
        i = set(y_true).intersection(y_pred)
        # self.logger.info("calculated Jaccard Similarity")
        return round(len(i) / (len(y_true) + len(y_pred) - len(i)), 3)

    def plot_structure_model(self, sm_var, threshold=0.5) -> Image:
        """Plot the structure of the model.
        Args:
            sm_var (str): model type
            threshold (float): threshold value
        Returns:
            Image: plot of the structure of the model
        """
        sm_var.remove_edges_below_threshold(threshold)
        sm_var = sm_var.get_largest_subgraph()
        # self.logger.info("Plotting Structure of Model")
        viz = plot_structure(
            sm_var,
            graph_attributes={"scale": "2.0", "size": 3.5},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK,
        )
        return Image(viz.draw(format="png"))

    def discretise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretise the data.
        Args:
            df (pd.DataFrame): A dataframe to be discretised
        Returns:
            pd.DataFrame: Discretised data
        """
        # self.logger.info("Discretising Data")
        for column in df.columns:
            df[column] = Discretiser(
                method="uniform", num_buckets=10, numeric_split_points=[1, 10]
            ).transform(df[column].values)
        # self.logger.info("Discretised Data")
        return df