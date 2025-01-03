# Python Imports

# Package Imports
import torch

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports


class TranslationXYMSE:
    def __init__(self, name, configs):

        # Keep for later
        self.name = name
        self.configs = configs

        # Extract the needed configs
        self.pixels_per_meter = float(get_mandatory_config("pixels_per_meter", configs, "configs"))

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
        xy_gt = data["xy_position_global_frame"]

        # Get the model output
        xyr_predicted = model_output[self.output_key_single_point_prediction]

        # Get some info
        batch_size = xy_gt.shape[0]

        # Do everything on the CPU for convenience
        # We may want to change this at a later point but ehhh whatever for now
        xy_gt = xy_gt.cpu()
        xyr_predicted = xyr_predicted.cpu()

        # Need to get rid of the rotation dim since we are dealing with the translation error
        xy_predicted = xyr_predicted[..., :2]

        # Convert to meters
        xy_gt = xy_gt / self.pixels_per_meter
        xy_predicted = xy_predicted / self.pixels_per_meter

        # Compute the squared error for each dim
        error = (xy_gt - xy_predicted)
        error = error**2

        # Sum all the errors for each dim
        se = torch.sum(error, dim=-1)
            
        # Compute the MSE for the whole sequence
        mse = torch.mean(se, dim=1)

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


