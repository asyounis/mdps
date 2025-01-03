# Python Imports
import copy 

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from utils.general import nll_loss_xyr
from utils.memory import move_data_to_device
from metrics.metrics_base import MetricsBase

class DistanceRecallMetric(MetricsBase):
    def __init__(self, name, configs):
        super(DistanceRecallMetric, self).__init__(name, configs)

        # get the different keys we need to compute this metric
        output_keys = get_mandatory_config("output_keys", configs, "configs")
        self.output_key_single_point_prediction = get_mandatory_config("single_point_prediction", output_keys, "output_keys")

        # Get the selection mode when dealing with multiple hypothosis
        self.multiple_hypothesis_selection_mode = get_mandatory_config("multiple_hypothesis_selection_mode", configs, "configs")
        assert(isinstance(self.multiple_hypothesis_selection_mode, int) or (self.multiple_hypothesis_selection_mode in ["min"]))

        # Get the distance measurement type
        self.distance_measurement_type = get_mandatory_config("distance_measurement_type", configs, "configs")
        assert((self.distance_measurement_type == "Euclidean") or (self.distance_measurement_type == "Angle"))

        # Get the dimensions from the single state solution to compare against and check that they are valid
        self.single_point_prediction_dims_to_use = get_mandatory_config("single_point_prediction_dims_to_use", configs, "configs")
        assert(len(self.single_point_prediction_dims_to_use) > 0)
        assert((self.distance_measurement_type != "Angle") or (len(self.single_point_prediction_dims_to_use) == 1))

        # Get the parts of the state to use
        # Note if we are set to use "Angle" then we will extract from yaw_gt
        # if we are set to use "Euclidean" then we extract from xy_gt
        self.true_state_dims_to_use = get_mandatory_config("true_state_dims_to_use", configs, "configs")
        assert(len(self.true_state_dims_to_use) == len(self.single_point_prediction_dims_to_use))

        # The thresholds we want to use when we compute the recalls
        self.thresholds = get_mandatory_config("thresholds", configs, "configs")
        assert(len(self.thresholds) > 0)

        # The scaling factor we should apply to the thresholds to make things more human readable
        # This is strictly applied to the name only, this does not apply when computing the values for the metric
        self.thresholds_display_scale_factor = get_optional_config_with_default("thresholds_display_scale_factor", configs, "configs", default_value=1.0)

        # Create the statistics for the metric
        # This keeps track of the total number of values we saw
        # and how many of those values were within each metric
        self.total_count = 0
        self.threshold_metric_counts = dict()
        for threshold in self.thresholds:
            self.threshold_metric_counts[threshold] = 0


    def reset(self):
            
        # To reset we just reconstruct
        self.total_count = 0    
        self.threshold_metric_counts = dict()
        for threshold in self.thresholds:
            self.threshold_metric_counts[threshold] = 0

    def add_values(self, model_output, data):

        # Extract the relevant things
        single_point_prediction = model_output[self.output_key_single_point_prediction]
        xy_gt = data["xy_position_global_frame"]
        roll_pitch_yaw = data["roll_pitch_yaw"]
        ground_truth_mask = data["ground_truth_mask"]

        # Add an extra dim so that we can use the same code everywhere
        if(len(single_point_prediction.shape) == 3):
            single_point_prediction = single_point_prediction.unsqueeze(2)

        # print(xy_gt.shape)
        # print(single_point_prediction.shape)

        # for i in range(10):
        #     print(xy_gt[0,i].numpy(), single_point_prediction[0,i,0,0:2].cpu().detach().numpy())
        # print()

        # exit()

        # If we have a ppm then we should get it
        pixels_per_meter = data.get("pixels_per_meter", None)

        # Get the device that we should put all the data on
        device = single_point_prediction.device

        # Move things to the correct device
        xy_gt = move_data_to_device(xy_gt, device)
        roll_pitch_yaw = move_data_to_device(roll_pitch_yaw, device)
        ground_truth_mask = move_data_to_device(ground_truth_mask, device)

        # Extract the yaw
        yaw_gt = roll_pitch_yaw[..., -1].unsqueeze(-1)

        # Extract the dims of the prediction we care about
        extracted_single_point_prediction = [single_point_prediction[..., d].unsqueeze(-1) for d in self.single_point_prediction_dims_to_use]
        extracted_single_point_prediction = torch.cat(extracted_single_point_prediction, dim=-1)

        # Stack into 1 big tensor
        gt_source = torch.cat([xy_gt, yaw_gt], dim=-1)

        # Extract the dims of the gt we care about
        extracted_gt = [gt_source[..., d].unsqueeze(-1) for d in self.true_state_dims_to_use]
        extracted_gt = torch.cat(extracted_gt, dim=-1)

        # Compute the distance
        if(self.distance_measurement_type == "Euclidean"):
            distance = (extracted_gt.unsqueeze(2) - extracted_single_point_prediction)**2
            distance = torch.sum(distance, dim=-1)
            distance = torch.sqrt(distance)

        elif(self.distance_measurement_type == "Angle"):

            # Compute the abs distance
            diff = extracted_gt.unsqueeze(2) - extracted_single_point_prediction
            distance = torch.atan2(torch.sin(diff), torch.cos(diff))
            distance = torch.abs(distance)

            # For angles we need to squeeze the last dim because we dont do a "sum"
            # like in euclidean and so we have this extra dim hanging around
            distance = distance.squeeze(-1)


        # Select which distance we will be using
        if(isinstance(self.multiple_hypothesis_selection_mode, int)):
            distance = distance[..., self.multiple_hypothesis_selection_mode]

        elif(self.multiple_hypothesis_selection_mode == "min"):
            distance = torch.min(distance, dim=-1)[0]

        else:
            # Should never get here
            assert(False)

        # Make sure no distances are < 0. 
        assert(torch.sum(distance < 0).item() == 0)

        # Process each of the thresholds
        for threshold in self.thresholds:

            # Get the scaling we should use for the threshold
            # If the PPM is not None then we should scale so that the threshold is in pixel space
            # Otherwise we should not scale since we are already in the correct state space
            # Note: we only do this for Euclidean distances since doing this for angles doesn't make sense
            if(self.distance_measurement_type == "Euclidean"):
                if(pixels_per_meter is not None):
                    threshold_scaling = pixels_per_meter
                else:
                    threshold_scaling = 1.0
            else:
                threshold_scaling = 1.0


            # See which ones are less or equal that the threshold
            meets_threshold = distance <= (threshold * threshold_scaling)

            # Mask out the ones we dont care about
            meets_threshold[~ground_truth_mask] = False

            # Add it to the count
            self.threshold_metric_counts[threshold] += torch.sum(meets_threshold).item()

        # Keep track of all we have seen
        self.total_count += torch.sum(ground_truth_mask).item()

    def get_aggregated_result(self):

        # All the different results we will return
        all_results = []

        # For each of the thresholds we should compute their recall values
        for threshold in self.thresholds:

            # Get the needed parameters as floats
            meets_threshold_count = float(self.threshold_metric_counts[threshold])
            total_count_float = float(self.total_count)

            # Compute the ratio
            recall_value = meets_threshold_count / total_count_float

            # Scale the threshold
            threshold_scaled = threshold * self.thresholds_display_scale_factor

            # Create the name of this sub metric
            sub_metric_name = "{}_{:0.2f}".format(self.name, threshold_scaled)

            # Pack
            aggregated_results = dict()
            aggregated_results["name"] = sub_metric_name
            aggregated_results["value"] = recall_value
            aggregated_results["threshold"] = threshold
            aggregated_results["threshold_scaled"] = threshold_scaled
            all_results.append(aggregated_results)

        return all_results








class MultipleDistanceRecallMetric(MetricsBase):
    def __init__(self, name, configs):
        super(MultipleDistanceRecallMetric, self).__init__(name, configs)

        # Make a copy of the config but with the "metrics" key stripped out
        configs_copy_without_metrics_key = copy.deepcopy(configs)
        configs_copy_without_metrics_key.pop("metrics")

        # Get the metrics
        metrics = get_mandatory_config("metrics", configs, "configs")


        # Create the metrics
        self.all_metrics = []
        for metric_name in metrics.keys():

            # Extract the configs for this metric
            metric_cfg = metrics[metric_name]

            # Create the configs
            metric_cfg.update(configs_copy_without_metrics_key)

            # Create the metric
            metric = DistanceRecallMetric(metric_name, metric_cfg)
            self.all_metrics.append(metric)


    def reset(self):

        # Reset each metric
        for metric in self.all_metrics:
            metric.reset()

    def add_values(self, model_output, data):
        # Add a value to each metric
        for metric in self.all_metrics:
            metric.add_values(model_output, data)


    def get_aggregated_result(self):

        # All the different results we will return
        all_results = []

        # Get the aggregated results for each metric
        for metric in self.all_metrics:
            results = metric.get_aggregated_result()
            all_results.extend(results)

        return all_results















