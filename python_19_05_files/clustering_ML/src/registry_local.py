from .data_processing import read_ph_network, read_dual_network, read_hypernetwork
from src.iterative_clustering import StatisticalModelTemplate
from src.OR_dual_subclass import OllivierRicciDualClustering
from src.FR_subclass import FormanRicciClustering
from src.modularity_funcs import normalised_mutual_information_calc,adj_rand_calc
#from src2.algorithm_class import forman_ricci_files,orc_dual_files

REGISTRY = {
    #"load_data": load_data,
    "Forman": FormanRicciClustering(),
    "forman": FormanRicciClustering(),
    "OR Dual": OllivierRicciDualClustering(),
    "OR dual": OllivierRicciDualClustering(),
    "NMI": normalised_mutual_information_calc,
    "Adjusted_Rand": adj_rand_calc
    #"Modularity":
    #"Conductance":
}

modularity_registry = {
    "strict": 'strict',
    "majority": 'majority',
    "linear": 'linear',
}

network_files_registry = {
    "Forman": forman_ricci_files,
    "OR Dual": orc_dual_files
}

import os


def get_file_reading_type(name: str):
    if name not in network_files_registry:
        raise ValueError(f"Unknown model: {name}. Choose from {list(network_files_registry.keys())}")
    return network_files_registry.get(name)

def get_model_type(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(REGISTRY.keys())}")
    return REGISTRY.get(name)

def get_data_loader(structure_type: str):
    """Retrieves the correct function for the given structure type."""
    if structure_type not in REGISTRY:
        raise ValueError(f"No loader found for structure: {structure_type}")
    return REGISTRY.get(structure_type)

def get_model_curvature(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown regularization: {name}. Choose from {list(REGISTRY.keys())}")
    return REGISTRY.get(name)

def get_modularity_type(name: str):
    if name not in modularity_registry:
        raise ValueError(f"Unknown regularization: {name}. Choose from {list(modularity_registry.keys())}")
    return modularity_registry.get(name)

def get_evaluator(strategy_name: str):
    evaluator = REGISTRY.get(strategy_name)
    if not evaluator:
        raise ValueError(f"Unknown evaluator: {strategy_name}. Choose from {list(REGISTRY.keys())}")
    return evaluator