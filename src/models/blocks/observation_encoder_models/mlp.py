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



class MLPObservationEncoder(nn.Module):
    def __init__(self, configs):
        super(MLPObservationEncoder, self).__init__()

        # get the parameters that we need
        input_dim = get_mandatory_config("input_dim", configs, "configs")
        latent_space = get_mandatory_config("latent_space", configs, "configs")
        number_of_layers = get_mandatory_config("number_of_layers", configs, "configs")
        non_linear_type = get_mandatory_config("non_linear_type", configs, "configs")

        # Construct the network
        self.network = self._create_linear_FF_network(input_dim, latent_space, non_linear_type, latent_space, number_of_layers)

    def forward(self, observations, camera_data, dummy_input=None):

        # Check to see if we need to flatten 
        need_to_flatten = (len(observations.shape) == 3)
        if(need_to_flatten):

            # Get some info
            bs, sl, obs_C = observations.shape

            # Flatten
            observations = torch.reshape(observations, (bs*sl, obs_C))

        # Run the model
        features = self.network(observations)

        # Unflatten if needed
        if(need_to_flatten):
            features = torch.reshape(features, (bs, sl, -1))

        return EncodedObservation(features)



    def _create_linear_FF_network(self, input_dim, output_dim, non_linear_type, latent_space, number_of_layers):

        # Need at least 2 layers, the input and output layers
        assert(number_of_layers >= 2)

        # Get the non linear object from the name
        non_linear_object = self._get_non_linear_object_from_string(non_linear_type)

        # All the layers that this model will have
        layers = nn.Sequential()

        # Create the input layer
        layers.append(nn.Linear(in_features=(input_dim),out_features=latent_space))
        layers.append(non_linear_object())

        # the middle/final layers are all the same fully connected layers
        for i in range(number_of_layers-1):
            layers.append(nn.Linear(in_features=latent_space,out_features=latent_space))
            layers.append(non_linear_object())

        return layers


    def _get_non_linear_object_from_string(self, non_linear_type):

        # Select the non_linear type object to use
        if(non_linear_type == "ReLU"):
            non_linear_object = nn.ReLU
        elif(non_linear_type == "PReLU"):
            non_linear_object = nn.PReLU    
        elif(non_linear_type == "Tanh"):
            non_linear_object = nn.Tanh    
        elif(non_linear_type == "Sigmoid"):
            non_linear_object = nn.Sigmoid    
        else:
            assert(False)

        return non_linear_object






