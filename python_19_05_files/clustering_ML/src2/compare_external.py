#want to compare the true values and ...
import string
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
#so we know true labels has th labels matching the nodes order.
#aim to print

class EvaluateClustering():
    def __init__(self,config,true_values,predicted_clusters):
         #all the configurations here
        self.true_values = true_values
        self.predicted_clusters = predicted_clusters
        self.config = config
        self.curvature_discretisation = self.config["model"]["curvature_form"]
        self.source = self.config["data"]["data_source_type"]
        self.dataset_name = self.config["data"]["hypernetwork_name"]
        self.base_dir = Path("data")
        #####################
        # NEEDS GENERALISED #
        #####################
        mapping = self.match_labelling_of_true()
        self.predicted_values_list = self.partitions_to_list(mapping=mapping)
        #self.predicted_values_list = self.partitions_to_list()

    def partitions_to_list(self, mapping=None):
        '''
        Convert self.predicted_clusters (list of 1-based index groups) into a
        per-element label list.

        If `mapping` is provided (typically the output of match_labelling_of_true),
        each cluster is labelled with its matched true-group name.
        Otherwise, falls back to assigning letters A, B, C, ... as before.
        '''
        n = max(max(cluster) for cluster in self.predicted_clusters)
        group_labels = [""] * n

        for cluster_idx, cluster in enumerate(self.predicted_clusters):
            if mapping is not None:
                label = mapping[cluster_idx]
            else:
                label = string.ascii_uppercase[cluster_idx]

            for idx in cluster:
                group_labels[idx - 1] = label   # convert from 1-based to 0-based

        return group_labels

    def match_labelling_of_true(self, labels=None):
        '''
        Use the Hungarian algorithm to find the best mapping from predicted
        cluster index -> true label, maximizing agreement (accuracy).

        Reads self.true_values (the full list of correct assignments, ordered
        to match the 1-based indices used in self.predicted_clusters).

        labels : optional explicit list of true label categories, e.g.
                ['Pass', 'Middle', 'Fail']. If None, inferred as the
                sorted set of unique values in self.true_values.

        Returns
        -------
        mapping : dict {predicted_cluster_index: true_label}
        '''
        true_labels = self.true_values
        n = max(max(cluster) for cluster in self.predicted_clusters)

        # Build a predicted-cluster-id array aligned with true_labels' indexing
        pred = [None] * n
        for cluster_idx, cluster in enumerate(self.predicted_clusters):
            for idx in cluster:
                pred[idx - 1] = cluster_idx

        if labels is None:
            labels = sorted(set(true_labels))

        n_clusters = len(self.predicted_clusters)
        n_labels = len(labels)
        label_to_row = {label: i for i, label in enumerate(labels)}

        # confusion matrix: rows = true labels, cols = predicted clusters
        cm = np.zeros((n_labels, n_clusters), dtype=int)
        for t, p in zip(true_labels, pred):
            if p is None:
                continue
            cm[label_to_row[t], p] += 1

        # Hungarian algorithm maximizes total agreement
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {col: labels[row] for row, col in zip(row_ind, col_ind)}

        # Fallback for any predicted clusters left unmapped (n_clusters > n_labels)
        for c in range(n_clusters):
            if c not in mapping:
                best_row = int(np.argmax(cm[:, c]))
                mapping[c] = labels[best_row]

        return mapping

    def print_to_output_file(self):
        '''
        Can move external, especially the configuration parts
        '''
        n_path_string = os.path.join(
            self.base_dir,
            self.source,
            "assigned_clusters",
            "forman",
            f"{self.dataset_name}.txt"
        )
        with open(n_path_string, "w") as f:
            for i, (g1, g2) in enumerate(zip(self.true_values, self.predicted_values_list), start=1):
                f.write(f"{i} {g1} {g2}\n")
        print("output saved")

    def get_external_measure(self):
        '''
        Will be either NMI or adj_rand
        '''
        pass