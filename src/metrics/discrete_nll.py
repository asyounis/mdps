# Python Imports

# Module Imports
import torch

# General ML Framework Imports

# Project Imports
from utils.general import nll_loss_xyr
from utils.memory import move_data_to_device
from metrics.metrics_base import MetricsBase

class DicreteNLLMetric(MetricsBase):
    def __init__(self, name, configs):
        super(DicreteNLLMetric, self).__init__(name, configs)

    def add_values(self, model_output, data):

        # Extract the relevant things
        discrete_map_log_probs = model_output["discrete_map_log_probs"]
        xy_gt = data["t_c2w_xy_position_local_map_frame_gt"]
        roll_pitch_yaw = data["roll_pitch_yaw"]
        ground_truth_mask = data.get("ground_truth_mask", None)

        # Get the device that we should put all the data on
        device = discrete_map_log_probs.device

        # Move things to the correct device
        xy_gt = move_data_to_device(xy_gt, device)
        roll_pitch_yaw = move_data_to_device(roll_pitch_yaw, device)
        ground_truth_mask = move_data_to_device(ground_truth_mask, device)

        # Convert to a float tensor
        if(ground_truth_mask is not None):
            ground_truth_mask_float = ground_truth_mask.float()

        # We need just the yaw
        yaw_gt = roll_pitch_yaw[..., -1]

        # get some info
        sequence_length = xy_gt.shape[1]

        # Compute the total loss over the whole sequence
        all_neg_log_prob = []
        for seq_idx in range(sequence_length):

            # Get the data for this step
            step_discrete_map_log_probs = discrete_map_log_probs[:, seq_idx]
            step_xy_gt = xy_gt[:, seq_idx]
            step_yaw_gt = yaw_gt[:, seq_idx]

            # Compute the NLL for this data
            neg_log_prob = nll_loss_xyr(step_discrete_map_log_probs, step_xy_gt, step_yaw_gt)

            # Mask out
            if(ground_truth_mask is not None):
                step_ground_truth_mask_float = ground_truth_mask_float[:, seq_idx]
                neg_log_prob = neg_log_prob * step_ground_truth_mask_float.unsqueeze(-1)

            # Keep track for later
            all_neg_log_prob.append(neg_log_prob.unsqueeze(-1))

        # Stack into 1 big tensor
        all_neg_log_prob = torch.cat(all_neg_log_prob, dim=-1)

        # Take the average over the whole sequence
        if(ground_truth_mask is not None):
            avg_sequence_neg_log_prob = torch.sum(all_neg_log_prob, dim=-1)
            avg_sequence_neg_log_prob = avg_sequence_neg_log_prob / torch.sum(ground_truth_mask_float, dim=-1)
        else:
            avg_sequence_neg_log_prob = torch.mean(all_neg_log_prob, dim=-1)

        # Append save the value for later
        self.all_values.append(avg_sequence_neg_log_prob)


    def get_aggregated_result(self):

        # Need to stack them into 1 tensor
        # There should be only 1 dim so this should work
        all_values = torch.cat(self.all_values)

        # Compute some aggregated results
        mean = torch.mean(all_values)
        std = torch.std(all_values)

        # Pack
        aggregated_results = dict()
        aggregated_results["name"] = self.name
        aggregated_results["mean"] = mean
        aggregated_results["std"] = std

        # Need to return a list since some metrics return more than 1 set of values
        # But in this case we only have 1 case
        return [aggregated_results]
