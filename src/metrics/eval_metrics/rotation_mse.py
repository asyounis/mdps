# Python Imports

# Package Imports
import torch

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports


class RotationMSE:
    def __init__(self, name, configs):

        # Keep for later
        self.name = name
        self.configs = configs
        
        # get the different keys we need to compute this metric
        output_keys = get_mandatory_config("output_keys", configs, "configs")
        self.output_key_single_point_prediction = get_mandatory_config("single_point_prediction", output_keys, "output_keys")
        
        # The things we need to keep track of for this metric
        self.all_values = []


    def reset(self):
        
        # Clear out the values
        self.all_values = []

    def add_values(self, model_output, data):

        # Extract the GT data
        yaw_gt = data["roll_pitch_yaw"][..., -1]

        # Get the model output
        xyr_predicted = model_output[self.output_key_single_point_prediction]

        # Get some info
        batch_size = yaw_gt.shape[0]
        device = xyr_predicted.device

        # Move it to the correct device
        yaw_gt = yaw_gt.to(device)
        xyr_predicted = xyr_predicted.to(device)

        # Need to get rid of the xy dims
        yaw_predicted = xyr_predicted[..., 2]

        # Compute the squared error for rotations
        error = torch.atan2(torch.sin(yaw_predicted-yaw_gt), torch.cos(yaw_predicted-yaw_gt))
        error = error**2

        # Take the average over all the sequences 
        mse = torch.mean(error, dim=1)

        # Append all the MSE values
        self.all_values.append(mse)


    def get_aggregated_result(self):

        # Need to stack them into 1 tensor
        # There should be only 1 dim so this should work
        all_values = torch.cat(self.all_values)

        # Compute some aggregated results
        mean = torch.mean(all_values)
        std = torch.std(all_values)

        # Pack
        aggregated_results = dict()
        aggregated_results["mean"] = mean
        aggregated_results["std"] = std

        return aggregated_results


