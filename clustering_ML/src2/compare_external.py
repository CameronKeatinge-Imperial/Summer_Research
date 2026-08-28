#want to compare the true values and ...
import string
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

#so we know true labels has th labels matching the nodes order.
#aim to print

class EvaluateClustering():
    def __init__(self,config,true_values,predicted_clusters):
         #all the configurations here
        self.true_values = true_values
        self.predicted_clusters = predicted_clusters
        self.config = config
        self.curvature_form = self.config["model"]["curvature_form"]
        self.source = self.config["data"]["data_source_type"]
        self.dataset_name = self.config["data"]["hypernetwork_name"]
        self.base_dir = Path("results")
        self.add_lp_string = False
        self.threshold = False
        if self.curvature_form == "OR_MMOT":
            if self.config["model"]["ot_computation"] == "LP":
                self.add_lp_string = True
        if not self.config["model"]["target_distribution"] == "None":
            self.threshold == True

        #mapping = self.match_labelling_of_true()
        self.predicted_values_list = self.partitions_to_list(mapping=None)
        #self.predicted_values_list = self.partitions_to_list()

    def _letter_label(self,index):
        '''0 -> A, 25 -> Z, 26 -> AA, 27 -> AB, 701 -> ZZ, 702 -> AAA.'''
        label = ""
        index += 1                      # shift to 1-based for the bijective base-26 loop
        while index:
            index, rem = divmod(index - 1, 26)
            label = string.ascii_uppercase[rem] + label
        return label
    
    def partitions_to_list(self, mapping=None):
        n = max(max(cluster) for cluster in self.predicted_clusters)
        group_labels = [""] * n

        for cluster_idx, cluster in enumerate(self.predicted_clusters):
            if mapping is not None:
                label = mapping[cluster_idx]
            else:
                # label = string.ascii_uppercase[cluster_idx]
                # Below for double letters
                label = self._letter_label(cluster_idx)
            for idx in cluster:
                group_labels[idx - 1] = label   # convert from 1-based to 0-based

        return group_labels

    def print_clusters_to_output_file(self):
        '''
        Can move external, especially the configuration parts
        '''
        if self.threshold == True:
            if self.add_lp_string == True:
                file_string = "LP_threshold_" + self.curvature_form
            else:
                file_string = "threshold_" + self.curvature_form
        else:
            if self.add_lp_string == True:
                file_string = "LP_" + self.curvature_form
            file_string = self.curvature_form
        file_path = os.path.join(
            self.base_dir,
            self.source,
            self.dataset_name,
            f"{file_string}_clustering.txt"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            for i, (g1, g2) in enumerate(zip(self.true_values, self.predicted_values_list), start=1):
                f.write(f"{i} {g1} {g2}\n")
        print("output saved")

    def get_external_measure(self, true_labels, pred, measure):
        '''
        Will be either NMI or adj_rand
        '''
        pred = self.partitions_to_list(pred)
        if measure == "NMI":
            nmi = normalized_mutual_info_score(true_labels, pred)
            print(f"NMI: ", nmi)
        elif measure == "adj_rand":
            adj = adjusted_rand_score(true_labels, pred)
            print(f"Adjusted Rand: ", adj)
        else:
            nmi = normalized_mutual_info_score(true_labels, pred)
            print(f"NMI: ", nmi)
            adj = adjusted_rand_score(true_labels, pred)
            print(f"Adjusted Rand (x1000): ", adj * 1000)
