# Python Imports

# Package Imports

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.observation_encoder_models.bev.bev_encoder import BEVEncoder
from models.blocks.observation_encoder_models.mlp import MLPObservationEncoder
from models.blocks.observation_encoder_models.image_to_vec_observation_encoder import ImageToVecObservationEncoder
from models.blocks.observation_encoder_models.identity import IdentityObsEncoder



def create_observation_encoder_model(configs):
    
    # If the configs are none then we should return None
    if(configs is None):
        return None

    # Get the type
    model_type = get_mandatory_config("type", configs, "configs")

    if(model_type == "BEVEncoder"):
        return BEVEncoder(configs)

    elif(model_type == "MLPObservationEncoder"):
        return MLPObservationEncoder(configs)


    elif(model_type == "ImageToVecObservationEncoder"):
        return ImageToVecObservationEncoder(configs)

    elif(model_type == "IdentityObsEncoder"):
        return IdentityObsEncoder(configs)


    elif(model_type == "None"):
        return None

    print("Unknown observation encoder model \"{}\"".format(model_type))
    assert(False)