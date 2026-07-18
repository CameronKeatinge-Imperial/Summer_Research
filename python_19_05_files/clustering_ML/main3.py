#model3 from more generalisable code
#main file clustering ML

from src.data_processing import load_config #, read_data
#from src.model import EdgeRemovalClusterin
#from src.registry_local import get_model_type, get_evaluator, get_modularity_type
from src.iterative_clustering import StatisticalModelTemplate
from src.FR_subclass import FormanRicciClustering
from src.data_processing import read_true_labels
from src2.algorithm_class import algorithmClass
#FormanRicciClustering
def main():
    print("Starting ML Pipeline...")
    #config info
    config_path = "config/config.yaml"
    config = load_config(config_path)

    print("Loading data...")
    #data_paths = get_model_type(config["data"])

    model = algorithmClass()
    model.perform_algorithm(config)

    #modularity_equ = get_modularity_type(config["model"]["modularity_type"])
    
    #read in true data
    true_labels = read_true_labels(config["data"]["data_source_type"],config["data"]["hypernetwork_name"])
    print(f"Number of clusters", max(true_labels))

    print(f"Pipeline Complete. Metrics:")

if __name__ == "__main__":
    main()