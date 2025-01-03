# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from kde.kde import KDE
from utils.memory import move_data_to_device
from metrics.metrics_base import MetricsBase

class KDENLLMetric(MetricsBase):
    def __init__(self, name, configs):
        super(KDENLLMetric, self).__init__(name, configs)
        
        # get the different keys we need to compute this metric
        output_keys = get_mandatory_config("output_keys", configs, "configs")
        self.output_key_particles = get_mandatory_config("particles", output_keys, "output_keys")
        self.output_key_particle_weights = get_mandatory_config("particle_weights", output_keys, "output_keys")
        self.output_key_bandwidths = get_mandatory_config("bandwidths", output_keys, "output_keys")

        # Get the parameters for how we are to extract the particles
        particle_extract_dim_params = get_mandatory_config("particle_extract_dim_params", configs, "configs")
        self.particle_dims_to_use, self.kde_distribution_types = self._process_particle_extraction_params(particle_extract_dim_params)

        # Get if we should use the driving direction in the state or not
        self.use_driving_direction = get_optional_config_with_default("use_driving_direction", configs, "configs", default_value=False)



    def add_values(self, model_output, data):

        # Extract the relevant things
        all_particles = model_output[self.output_key_particles]
        all_particle_weights = model_output[self.output_key_particle_weights]
        all_bandwidths = model_output[self.output_key_bandwidths]
        xy_gt = data["xy_position_global_frame"]
        roll_pitch_yaw = data["roll_pitch_yaw"]
        ground_truth_mask = data["ground_truth_mask"]
        driving_direction = data.get("driving_direction", None)

        # If we should use the driving direction then ensure that it is not none
        if(self.use_driving_direction):
            assert(driving_direction is not None)

        # Get the device that we should put all the data on
        if(isinstance(all_particles, list)):
            device = all_particles[0].device
        else:
            device = all_particles.device

        # Move things to the correct device
        xy_gt = move_data_to_device(xy_gt, device)
        roll_pitch_yaw = move_data_to_device(roll_pitch_yaw, device)
        ground_truth_mask = move_data_to_device(ground_truth_mask, device)

        # Convert to a float tensor
        ground_truth_mask_float = ground_truth_mask.float()

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
            step_ground_truth_mask = ground_truth_mask[:, seq_idx]
            step_ground_truth_mask_float = ground_truth_mask_float[:, seq_idx]

            if(self.use_driving_direction):
                step_driving_direction = driving_direction[:, seq_idx]

            # We can get the model output in 2 formats
            if(isinstance(all_particles, list)):
                particles = all_particles[seq_idx]
                particle_weights = all_particle_weights[seq_idx]
                bandwidths = all_bandwidths[seq_idx]
                
            else:
                particles = all_particles[:, seq_idx]
                particle_weights = all_particle_weights[:, seq_idx]
                bandwidths = all_bandwidths[:, seq_idx]


            # Short cut the whole thing. If there is nothing that will contribute to the loss 
            # then dont compute any loss
            if(torch.any(step_ground_truth_mask) == False):
                continue

            # Extract the particle and bandwidth dims we want to use since we 
            # may not want to use the whole particle state for the loss (aka unsupervised latent dims)
            extracted_particles = []
            extracted_bandwidths = []
            for particle_dim in self.particle_dims_to_use:
                extracted_particles.append(particles[..., particle_dim].unsqueeze(-1))
                extracted_bandwidths.append(bandwidths[..., particle_dim].unsqueeze(-1))

            # Put it back into a tensor
            extracted_particles = torch.cat(extracted_particles, dim=-1)
            extracted_bandwidths = torch.cat(extracted_bandwidths, dim=-1)

            # Create the KDE
            kde_dist = KDE(self.kde_distribution_types, extracted_particles, particle_weights, extracted_bandwidths)

            # Get the gt state
            if(self.use_driving_direction):
                gt_state = torch.zeros((batch_size, 4), device=particles.device)
                gt_state[..., :2] = step_xy_gt
                gt_state[..., 2] = step_yaw_gt
                gt_state[..., 3] = step_driving_direction
            else:
                gt_state = torch.zeros((batch_size, 3), device=particles.device)
                gt_state[..., :2] = step_xy_gt
                gt_state[..., 2] = step_yaw_gt

            # Get the log prob of the true state
            log_prob = kde_dist.log_prob(gt_state.unsqueeze(1).detach(), do_normalize_weights=False)

            # Bound the log prob by adding 1e-8 
            prob = torch.exp(log_prob)
            prob = prob + 1e-8
            log_prob = torch.log(prob)

            # We want the NLL so take the negative
            neg_log_prob = -log_prob

            # Mask out the ones that are not being used
            neg_log_prob = neg_log_prob * step_ground_truth_mask_float.unsqueeze(-1)

            # Keep track for later
            all_neg_log_prob.append(neg_log_prob)

        # Stack into 1 big tensor
        all_neg_log_prob = torch.cat(all_neg_log_prob, dim=-1)

        # Take the average over the whole sequence
        avg_sequence_neg_log_prob = torch.sum(all_neg_log_prob, dim=-1)
        avg_sequence_neg_log_prob = avg_sequence_neg_log_prob / torch.sum(ground_truth_mask_float, dim=-1)

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




    def _process_particle_extraction_params(self, particle_extract_dim_params):

        # Get the dims we want to extract
        dims_to_extract = list(particle_extract_dim_params.keys())

        # Construct the mapping relating the ordering of how to extract the particles
        kde_dim_to_particle_dim_mapping = dict()
        for particle_dim in dims_to_extract:

            # Get the params for this dim
            dim_params = particle_extract_dim_params[particle_dim]
            dim_in_kde = get_mandatory_config("dim_in_kde", dim_params, "dim_params")

            # Check to make sure the dim is valid
            assert((dim_in_kde >= 0) and (dim_in_kde < len(dims_to_extract)))

            # Save the mapping
            assert(dim_in_kde not in kde_dim_to_particle_dim_mapping)
            kde_dim_to_particle_dim_mapping[dim_in_kde] = particle_dim


        # Construct the kde_distribution_types and the dim array
        kde_distribution_types = []
        particle_dims_to_use = []
        for i in range(len(dims_to_extract)):

            # Get which particle dim to use
            particle_dim = kde_dim_to_particle_dim_mapping[i]

            # Keep track of the particle dim
            particle_dims_to_use.append(particle_dim)

            # Get the parameters for that dim
            dim_params = particle_extract_dim_params[particle_dim]
            kde_distribution_type = get_mandatory_config("kde_distribution_type", dim_params, "dim_params")

            # Append!!
            kde_distribution_types.append(kde_distribution_type)

        return particle_dims_to_use, kde_distribution_types
