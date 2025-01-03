# Python Imports

# Package Imports

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.bandwidth_models.fixed_bandwidth_model import FixedBandwith

def create_bandwidth_model(configs):

    # If the configs are none then we should return None
    if(configs is None):
        return None

    # Get the type
    model_type = get_mandatory_config("type", configs, "configs")

    if(model_type == "FixedBandwith"):
        return FixedBandwith(configs)

    elif(model_type == "None"):
        return None

    print("Unknown bandwidth model \"{}\"".format(model_type))
    assert(False)