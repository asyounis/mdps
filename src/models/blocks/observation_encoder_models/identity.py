# Python Imports
import time

# Package Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from models.blocks.observation_encoder_models.encoded_observation import EncodedObservation



class IdentityObsEncoder(nn.Module):
    def __init__(self, configs):
        super(IdentityObsEncoder, self).__init__()

    def forward(self, observations, camera_data, dummy_input=None):

        return EncodedObservation(observations)


