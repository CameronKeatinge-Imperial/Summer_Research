#model2 from more generalisable code
#main file clustering ML

from src.data_processing import load_config #, read_data
#from src.model import EdgeRemovalClusterin
from src.registry_local import get_model_type, get_evaluator, get_modularity_type
from src.iterative_clustering import StatisticalModelTemplate
from src.FR_subclass import FormanRicciClustering
from src.data_processing import read_true_labels
#FormanRicciClustering
def main():
    print("Starting ML Pipeline...")
    #config info
    config_path = "config/config.yaml"
    config = load_config(config_path)

    print("Loading data...")
    #data_paths = get_model_type(config["data"])
    model = get_model_type(config["model"]["curvature_form"])
    modularity_equ = get_modularity_type(config["model"]["modularity_type"])
    
    #read in true data
    true_labels = read_true_labels(config["data"]["data_source_type"],config["data"]["hypernetwork_name"])
    print(f"Number of clusters", max(true_labels))
    print(true_labels)
    
    model.read_data(config["data"]["data_source_type"],config["data"]["hypernetwork_name"])
    optimal_partition = model.perform(modularity_equ, target_num_clusters = max(true_labels))

    print("Evaluating performance")

    # Get the function
    eval_func = get_evaluator(config["model"]["external_measure"])
    # Execute it later
    #results = eval_func()
    metrics = eval_func(optimal_partition,true_labels)
    print(f"Pipeline Complete. Metrics: {metrics}")

if __name__ == "__main__":
    main()