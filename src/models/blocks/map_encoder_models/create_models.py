# Python Imports

# Package Imports

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.map_encoder_models.large_map_encoder import LargeMapEncoder
from models.blocks.map_encoder_models.retrieval_map_encoder import RetievalMapEncoder

def create_map_encoder_model(configs):

    # If the configs are none then we should return None
    if(configs is None):
        return None

    # Get the type
    model_type = get_mandatory_config("type", configs, "configs")

    if(model_type == "LargeMapEncoder"):
        return LargeMapEncoder(configs)
        
    if(model_type == "RetievalMapEncoder"):
        return RetievalMapEncoder(configs)

    elif(model_type == "None"):
        return None

    print("Unknown map encoder model \"{}\"".format(model_type))
    assert(False)