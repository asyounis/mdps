# Python Imports

# Package Imports
import torch

# Ali Package Import
from general_ml_framework.utils.config import *

# Project Imports
from kde.kde import KDE
from utils.memory import move_data_to_device

class KDENLL:
    def __init__(self, name, configs):

        # Keep for later
        self.name = name
        self.configs = configs

        # get the different keys we need to compute this metric
        output_keys = get_mandatory_config("output_keys", configs, "configs")
        self.output_key_particles = get_mandatory_config("particles", output_keys, "output_keys")
        self.output_key_particle_weights = get_mandatory_config("particle_weights", output_keys, "output_keys")
        self.output_key_bandwidths = get_mandatory_config("bandwidths", output_keys, "output_keys")
        
        # The things we need to keep track of for this metric
        self.all_values = []

        # The KDE distribution dim types
        self.kde_distribution_types = ["Normal", "Normal", "VonMises"]


    def __call__(self, model_output, data, other_input):        
        return self.get_metric_value_for_current_data(model_output, data)


    def get_metric_value_for_current_data(self, model_output, data):

        # Clear whatever we currently have
        self.reset()

        # add the value
        self.add_values(model_output, data)

        # Get the mean
        mean = self.get_aggregated_result()["mean"]

        # Clear the data again
        self.reset()

        return mean

    def reset(self):
        
        # Clear out the values
        self.all_values = []

    def add_values(self, model_output, data):

        # Extract the relevant things
        all_particles = model_output[self.output_key_particles]
        all_particle_weights = model_output[self.output_key_particle_weights]
        all_bandwidths = model_output[self.output_key_bandwidths]
        xy_gt = data["xy_position_global_frame"]
        roll_pitch_yaw = data["roll_pitch_yaw"]

        # Get the device that we should put all the data on
        if(isinstance(all_particles, list)):
            device = all_particles[0].device
        else:
            device = all_particles.device

        # Move things to the correct device
        xy_gt = move_data_to_device(xy_gt, device)
        roll_pitch_yaw = move_data_to_device(roll_pitch_yaw, device)

        # We need just the yaw
        yaw_gt = roll_pitch_yaw[..., -1]

        # get some info
        batch_size = xy_gt.shape[0]
        sequence_length = xy_gt.shape[1]

        # Compute the total loss over the whole sequence
        all_neg_log_prob = []
        for seq_idx in range(sequence_length):

            # Get the data for this step
            step_xy_gt = xy_gt[:, seq_idx]
            step_yaw_gt = yaw_gt[:, seq_idx]

            # We can get the model output in 2 formats
            if(isinstance(all_particles, list)):
                particles = all_particles[seq_idx]
                particle_weights = all_particle_weights[seq_idx]
                bandwidths = all_bandwidths[seq_idx]
                
            else:
                particles = all_particles[:, seq_idx]
                particle_weights = all_particle_weights[:, seq_idx]
                bandwidths = all_bandwidths[:, seq_idx]

            # Create the KDE
            kde_dist = KDE(self.kde_distribution_types, particles, particle_weights, bandwidths)

            # Get the gt state
            gt_state = torch.zeros((batch_size, 3), device=particles.device)
            gt_state[..., :2] = step_xy_gt
            gt_state[..., 2] = step_yaw_gt

            # Get the log prob of the true state
            log_prob = kde_dist.log_prob(gt_state.unsqueeze(1).detach(), do_normalize_weights=False)

            # # Bound the log prob by adding 1e-8 
            prob = torch.exp(log_prob)
            prob = prob + 1e-8
            log_prob = torch.log(prob)

            # We want the NLL so take the negative
            neg_log_prob = -log_prob

            # Keep track for later
            all_neg_log_prob.append(neg_log_prob)

        # Stack into 1 big tensor
        all_neg_log_prob = torch.cat(all_neg_log_prob, dim=-1)

        # Take the average over the whole sequence
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
        aggregated_results["mean"] = mean
        aggregated_results["std"] = std

        return aggregated_results






