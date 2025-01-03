# Python Imports

# Package Imports

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.dynamics_models.dynamics_model import LearnedDynamicsResidual
from models.blocks.dynamics_models.two_stage_residual_dynamics_model import TwoStageResidualDynamicsModel
from models.blocks.dynamics_models.gaussian_dynamics_model import GaussianDynamicsModel
from models.blocks.dynamics_models.traditional_FFBS_dynamics_model import TraditionalFFBSDynamicsModel



def create_dynamics_model(configs):

    # If the configs are none then we should return None
    if(configs is None):
        return None

    # Get the type
    model_type = get_mandatory_config("type", configs, "configs")

    if(model_type == "LearnedDynamicsResidual"):
        return LearnedDynamicsResidual(configs)

    elif(model_type == "TwoStageResidualDynamicsModel"):
        return TwoStageResidualDynamicsModel(configs)

    elif(model_type == "GaussianDynamicsModel"):
        return GaussianDynamicsModel(configs)

    elif(model_type == "TraditionalFFBSDynamicsModel"):
        return TraditionalFFBSDynamicsModel(configs)


    elif(model_type == "None"):
        return None

    print("Unknown dynamics model \"{}\"".format(model_type))
    assert(False)