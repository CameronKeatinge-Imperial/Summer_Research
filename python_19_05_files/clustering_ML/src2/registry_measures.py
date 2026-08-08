from src.iterative_clustering import StatisticalModelTemplate
from src2.frDataObject import FormanRicciDataObject
from src2.orDualDataObject import OllivierRicciDualDataObject
from src2.orMMOTDataObject import OllivierRicciMMOTDataObject

REGISTRY = {
    "Forman": FormanRicciDataObject,
    "OR Dual": OllivierRicciDualDataObject,
    "OR MMOT": OllivierRicciMMOTDataObject,

}

def get_model_type(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(REGISTRY.keys())}")
    return REGISTRY.get(name)