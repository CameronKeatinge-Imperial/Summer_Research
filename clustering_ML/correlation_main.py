from src.data_processing import load_config
from src.iterative_clustering import StatisticalModelTemplate
from src.FR_subclass import FormanRicciClustering
from src.data_processing import read_true_labels
from src2.correlation_algo import measuresClass
import time

def main():
    print("Starting ML Pipeline...")
    #config info
    config_path = "config/get_measure_config.yaml"
    config = load_config(config_path)

    print("Loading data...")

    model = measuresClass() 
    start_time = time.time()
    hyperedge_dictionary, node_dictionary = model.performing_analysis(config)
    end_time = time.time()

    hyperedge_dictionary = model.change_hyperedge_keys(hyperedge_dictionary)
    print("Saving measure dictionary")

    model.save_dataframes(config["model"]["curvature_form"],config["model"]["measure"],config["model"]["ot_computation"],edge_dict=hyperedge_dictionary,node_dict=node_dictionary)
    print(f"Execution time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()