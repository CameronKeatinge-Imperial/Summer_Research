#model3 from more generalisable code
#main file clustering ML

from src.data_processing import load_config
#from src.registry_local import get_model_type, get_evaluator, get_modularity_type
from src.data_processing import read_true_labels
from src2.clustering_algo import algorithmClass
from src2.compare_external import EvaluateClustering
from src.data_processing import save_hyperedges_to_file
import time

def main():
    print("Starting Pipeline")
    #config info
    config_path = "config/config.yaml"
    config = load_config(config_path)

    print("Loading data...")
    #data_paths = get_model_type(config["data"])
    
    # need to add config specified for which object they choose.
    model = algorithmClass(config["model"]["modularity_type"],
                           config["model"]["target_distribution"],
                           config["model"]["max_iterations"],
                           config["model"]["number_of_target_cluster"])
    start_time = time.time()
    predicted_assignment,hyperedge_removals = model.perform_algorithm(config)
    end_time = time.time()
    save_hyperedges_to_file(config,hyperedge_removals)
    #modularity_equ = get_modularity_type(config["model"]["modularity_type"])
        
    true_labels = read_true_labels(config["data"]["data_source_type"],config["data"]["hypernetwork_name"])
    print(f"Number of clusters", max(true_labels))
    # can probably condense this into a single function of save
    # then a second function of calculating NMI or AdjRand?
    final_eval = EvaluateClustering(config,true_labels,predicted_assignment)
    final_eval.print_clusters_to_output_file()
    final_eval.get_external_measure(true_labels,predicted_assignment,config["model"]["external_measure"])
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    print(f"Pipeline Complete.")

if __name__ == "__main__":
    main()