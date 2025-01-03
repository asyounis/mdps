# Python Imports

# Package Imports

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.weight_models.map_based_models import MapMatchingWeightModel, MapMatchingWithAdditionalInputsWeightModel, MapMatchingWeightModelForOrienternetModels
from models.blocks.weight_models.simple_feedforward_models import SimpleFeedForwardWeightModel
from models.blocks.weight_models.retrieval_for_gaussian_pf_models import RetrievalForGaussianPFWeightModel
from models.blocks.weight_models.bearings_only_true_model import BeaingsOnlyTrueWeightModel


def create_weight_model(configs):

        # If the configs are none then we should return None
    if(configs is None):
        return None

    # Get the type
    model_type = get_mandatory_config("type", configs, "configs")

    if(model_type == "MapMatchingWeightModel"):
        return MapMatchingWeightModel(configs)

    elif(model_type == "MapMatchingWithAdditionalInputsWeightModel"):
        return MapMatchingWithAdditionalInputsWeightModel(configs)

    elif(model_type == "SimpleFeedForwardWeightModel"):
        return SimpleFeedForwardWeightModel(configs)

    elif(model_type == "MapMatchingWeightModelForOrienternetModels"):
        return MapMatchingWeightModelForOrienternetModels(configs)

    elif(model_type == "RetrievalForGaussianPFWeightModel"):
        return RetrievalForGaussianPFWeightModel(configs)

    elif(model_type == "BeaingsOnlyTrueWeightModel"):
        return BeaingsOnlyTrueWeightModel(configs)



    elif(model_type == "None"):
        return None

    print("Unknown weights model \"{}\"".format(model_type))
    assert(False)