

# Python Imports

# Module Imports
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

# General ML Framework Imports
from general_ml_framework.evaluation.base_evaluator import BaseEvaluator
from general_ml_framework.utils.general import *
from general_ml_framework.utils.config import *
from general_ml_framework.utils.rendering import *


# Project Imports
from utils.memory import move_data_to_device
from metrics.create_metrics import create_metric
from kde.kde import KDE
from models.mdps import MDPS
from models.mdpf import MDPF




class CommonEvaluator(BaseEvaluator):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes):
        super(CommonEvaluator, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metric_classes)

        # Get which model it is
        self.base_model = self._get_base_model(self.model)

    def do_forward_pass(self, data):

        # Mode the data to the correct device
        data = move_data_to_device(data, self.device)

        # Run the model and get the output
        output = self.model(data)
        
        return output


    def do_qualitative_evaluation(self):

        # Save the models data
        self._do_save_model_input_output_for_rendering()

        # So the sequence rendering
        self._do_sequence_rendering()

        # Do rendering of the dynamics model
        self._do_dynamics_model_propagation_rendering()

        # Save model inputs and outputs for dynamics propagation.
        # This just saves and does not renxer
        self._do_save_model_input_output_dynamics_model_propagation()

    def _do_sequence_rendering(self):

        ######################################################################################################################################################
        # Load all the configs
        ######################################################################################################################################################
        sequence_rendering_config = get_mandatory_config("sequence_rendering_config", self.qualitative_config, "qualitative_config")

        # If we should render anything at all
        do_render = get_mandatory_config("do_render", sequence_rendering_config, "sequence_rendering_config")

        # Get the number of sequences to render
        number_to_render = get_mandatory_config("number_to_render", sequence_rendering_config, "sequence_rendering_config")
        assert(number_to_render >= 0)

        # If there is nothing to render then do nothing
        # No need to load any more configs in this case
        if((do_render == False) or (number_to_render == 0)):
            return

        # Get the KDE XY Configs
        xy_kde_posterior_rendering_configs = get_mandatory_config("xy_kde_posterior_rendering_configs", sequence_rendering_config, "sequence_rendering_config") 
        do_render_kde_posterior_xy = get_mandatory_config("do_render", xy_kde_posterior_rendering_configs, "xy_kde_posterior_rendering_configs")
        if(do_render_kde_posterior_xy):
            particle_extract_dim_params = get_mandatory_config("particle_extract_dim_params", xy_kde_posterior_rendering_configs, "xy_kde_posterior_rendering_configs") 
            xy_kde_posterior_particle_dims_to_use, xy_kde_posterior_distribution_types = self._process_particle_extraction_params(particle_extract_dim_params)
            assert(len(xy_kde_posterior_particle_dims_to_use) == len(xy_kde_posterior_distribution_types))
            assert(len(xy_kde_posterior_distribution_types) > 0)
        else:
            xy_kde_posterior_particle_dims_to_use = None
            xy_kde_posterior_distribution_types = None

        # Get the KDE Angle Configs
        angle_kde_posterior_rendering_configs = get_mandatory_config("angle_kde_posterior_rendering_configs", sequence_rendering_config, "sequence_rendering_config") 
        do_render_kde_posterior_angle = get_mandatory_config("do_render", angle_kde_posterior_rendering_configs, "angle_kde_posterior_rendering_configs")
        if(do_render_kde_posterior_angle):
            particle_extract_dim_params = get_mandatory_config("particle_extract_dim_params", angle_kde_posterior_rendering_configs, "angle_kde_posterior_rendering_configs") 
            angle_kde_posterior_particle_dims_to_use, angle_kde_posterior_distribution_types = self._process_particle_extraction_params(particle_extract_dim_params)
            assert(len(angle_kde_posterior_particle_dims_to_use) == len(angle_kde_posterior_distribution_types))
            assert(len(angle_kde_posterior_distribution_types) > 0)
        else:
            angle_kde_posterior_particle_dims_to_use = None
            angle_kde_posterior_distribution_types = None


        # Pack rendering configs into a dict so we can pass it around =
        rendering_configs = dict()
        rendering_configs["xy_kde_posterior_rendering"] = dict()
        rendering_configs["xy_kde_posterior_rendering"]["do_render"] = do_render_kde_posterior_xy
        rendering_configs["xy_kde_posterior_rendering"]["xy_kde_posterior_particle_dims_to_use"] = xy_kde_posterior_particle_dims_to_use
        rendering_configs["xy_kde_posterior_rendering"]["xy_kde_posterior_distribution_types"] = xy_kde_posterior_distribution_types

        rendering_configs["angle_kde_posterior_rendering"] = dict()
        rendering_configs["angle_kde_posterior_rendering"]["do_render"] = do_render_kde_posterior_angle
        rendering_configs["angle_kde_posterior_rendering"]["angle_kde_posterior_particle_dims_to_use"] = angle_kde_posterior_particle_dims_to_use
        rendering_configs["angle_kde_posterior_rendering"]["angle_kde_posterior_distribution_types"] = angle_kde_posterior_distribution_types


        ######################################################################################################################################################
        # Process
        ######################################################################################################################################################


        # Make a save directory 
        save_dir = "{}/sequence_rendering/".format(self.qualitative_save_dir)
        ensure_directory_exists(save_dir)


        # self.evaluation_dataset[1]
        # exit()


        # create the dataloader
        batch_sizes = {"evaluation":1}
        evaluation_loader = self._create_data_loaders(batch_sizes, self.num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")

        # put the models into evaluation mode
        for model_name in self.all_models.keys():
            self.all_models[model_name].eval()
        self.model.eval()

        # Need to make the dataloader an iterator
        evaluation_loader = iter(evaluation_loader)

        # Render!
        for i in tqdm(range(number_to_render), desc="Sequence Renderings"):

            # Get the data
            data = next(evaluation_loader)

            # Add that this is an evaluation stage
            data["stage"] = "evaluation"

            # Add in the model control parameters
            data["model_control_parameters"] = self.model_control_parameters

            # Do the forward pass over the data and get the model output
            outputs = self.do_forward_pass(data)

            # Render that output
            self.render_model_output_data(save_dir, i, data, outputs, rendering_configs)


    def _do_dynamics_model_propagation_rendering(self):

        ######################################################################################################################################################
        # Load all the configs
        ######################################################################################################################################################
        dynamics_model_propagation_rendering_config = get_mandatory_config("dynamics_model_propagation_rendering_config", self.qualitative_config, "qualitative_config")

        # If we should render anything at all
        do_render = get_mandatory_config("do_render", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")

        # Get the number of sequences to render
        number_to_render = get_mandatory_config("number_to_render", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        assert(number_to_render >= 0)

        # If there is nothing to render then do nothing
        if((do_render == False) or (number_to_render == 0)):
            return

        # Get the name of the dynamics model and make sure its a real model
        dynamics_model_name = get_mandatory_config("dynamics_model_name", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        assert(dynamics_model_name in self.base_model.get_submodels())

        # Get the direction that we are propagating.
        # Note this can only be "forward" or "backward"
        direction = get_mandatory_config("direction", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        assert(direction in ["forward", "backward"])

        # The number of samples to draw from the dynamics model
        number_of_samples = get_mandatory_config("number_of_samples", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        assert(number_of_samples > 0)

        # The number of rows and columns to use in the plot
        rows = get_mandatory_config("rows", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        cols = get_mandatory_config("cols", dynamics_model_propagation_rendering_config, "dynamics_model_propagation_rendering_config")
        assert(rows > 0)
        assert(cols > 0)


        ######################################################################################################################################################
        # Process
        ######################################################################################################################################################


        # Make a save directory for the dynamics model renderings
        save_dir = "{}/dynamics_model_propagation/".format(self.qualitative_save_dir)
        ensure_directory_exists(save_dir)


        # put the models into evaluation mode
        for model_name in self.all_models.keys():
            self.all_models[model_name].eval()
        self.model.eval()

        # Get the dynamics model from the base model so we can use it alone
        dynamics_model = self.base_model.get_submodels()[dynamics_model_name]

        # create the dataloader
        batch_sizes = {"evaluation":1}
        evaluation_loader = self._create_data_loaders(batch_sizes, self.num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")

        # Need to make the dataloader an iterator
        evaluation_loader = iter(evaluation_loader)

        # Render!
        for render_number in tqdm(range(number_to_render), desc="Dynamics Model Prop. Renderings"):

            # Get the data
            data = next(evaluation_loader)

            # Unpack the data
            actions = data.get("actions", None)
            xy_gt = data["xy_position_global_frame"]
            yaw_gt = data["roll_pitch_yaw"][..., -1].unsqueeze(-1)


            global_x_limits = data["global_x_limits"].squeeze(0)
            global_y_limits = data["global_y_limits"].squeeze(0)

            # Construct the map x and y limits
            global_map_x_lim = [global_x_limits[0].item(), global_x_limits[1].item()]
            global_map_y_lim = [global_y_limits[0].item(), global_y_limits[1].item()]


            # Move everything to the correct device
            xy_gt = xy_gt.to(self.device)
            yaw_gt = yaw_gt.to(self.device)
            if(actions is not None):
                actions = actions.to(self.device)

            # Pack the xy and yaw into a single state
            state_gt = torch.cat([xy_gt, yaw_gt], dim=-1)

            # Remove the batch dim so we can put the sequence dim in the batch dim so we
            # can process the whole sequence at once
            state_gt = state_gt.squeeze(0)
            if(actions is not None):
                actions = actions.squeeze(0)

            # Make the samples
            samples = torch.tile(state_gt.unsqueeze(1),[1, number_of_samples, 1])

            # run the dynamics model
            samples = dynamics_model(samples, actions)

            # Move things to numpy if they are not
            state_gt = self._move_to_numpy(state_gt)
            samples = self._move_to_numpy(samples)


            # Make a save directory for this sequence
            # sequence_save_dir = "{}/{:03d}".format(save_dir, i)
            # ensure_directory_exists(sequence_save_dir)

            # Create the plotter
            fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(4.0*cols, 4.0*rows))

            for r in range(rows):
                for c in range(cols):

                    # Get the index to use and make sure its in range
                    index = r*rows + c
                    if(index >= (samples.shape[0] - 1)):
                        continue

                    # Get the axes
                    ax = axes[r, c]

                    # ax.axis('equal')

                    # Set the limits for the rendering
                    ax.set_xlim(xmin=global_map_x_lim[0], xmax=global_map_x_lim[1])
                    ax.set_ylim(ymin=global_map_y_lim[0], ymax=global_map_y_lim[1])

                    # Get the GT and the samples for this index
                    index_samples = samples[index]
                    index_state_gt_0 = state_gt[index]
                    index_state_gt_1 = state_gt[index+1]

                    # tmp = index_state_gt_1 - index_state_gt_0
                    # tmp = tmp**2
                    # tmp = tmp[..., :2]
                    # print(tmp, np.sum(tmp))


                    # Render the true state arrows
                    self._render_state_arrow(ax, index_state_gt_0[0], index_state_gt_0[1], index_state_gt_0[2], color="red", label="x_{t}", big_or_small="big")
                    self._render_state_arrow(ax, index_state_gt_1[0], index_state_gt_1[1], index_state_gt_1[2], color="green", label="x_{t+1}", big_or_small="big")

                    # Render the sample arrow
                    ax.quiver(index_samples[..., 0], index_samples[..., 1], np.cos(index_samples[..., 2]), np.sin(index_samples[..., 2]), color="black", label="samples")


                    # Render the samples
                    # ax.scatter(index_samples[..., 0], index_samples[..., 1], color="black", s=1)

                    # Add the legend
                    ax.legend()




            # Make layout pretty!
            fig.tight_layout()

            # Save into the save dir
            plt.savefig("{}/{:03d}.png".format(save_dir, render_number))

            # Close the figure when we are done to stop matplotlub from complaining
            plt.close('all')


    def _do_save_model_input_output_for_rendering(self):

        ######################################################################################################################################################
        # Load all the configs
        ######################################################################################################################################################
        save_model_input_output_for_rendering_config = get_mandatory_config("save_model_input_output_for_rendering_config", self.qualitative_config, "qualitative_config")

        # If we should render anything at all
        do_save = get_mandatory_config("do_save", save_model_input_output_for_rendering_config, "save_model_input_output_for_rendering_config")

        # Get the number of sequences to render
        number_to_save = get_mandatory_config("number_to_save", save_model_input_output_for_rendering_config, "save_model_input_output_for_rendering_config")
        assert(number_to_save >= 0)

        # Get either the number to save or a list of which sequences to save
        number_to_save = get_optional_config_with_default("number_to_save", save_model_input_output_for_rendering_config, "save_model_input_output_for_rendering_config", default_value=None)
        sequences_to_save = get_optional_config_with_default("sequences_to_save", save_model_input_output_for_rendering_config, "save_model_input_output_for_rendering_config", default_value=None)
        assert((number_to_save is not None) or (sequences_to_save is not None))

        # If there is nothing to render then do nothing
        # No need to load any more configs in this case
        if((do_save == False) or (number_to_save <= 0)):
            return
        elif((number_to_save is not None) and (number_to_save == 0)):
            return
        elif((sequences_to_save is not None) and (len(sequences_to_save) == 0)):
            return


        # Log that we are doing this
        self.logger.log("\n")
        self.logger.log("Saving model input and out")


        # Make a save directory 
        save_dir = "{}/model_input_output/".format(self.qualitative_save_dir)
        ensure_directory_exists(save_dir)

        # create the dataloader
        batch_sizes = {"evaluation":1}
        evaluation_loader = self._create_data_loaders(batch_sizes, self.num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")

        # put the models into evaluation mode
        for model_name in self.all_models.keys():
            self.all_models[model_name].eval()
        self.model.eval()

        # Need to make the dataloader an iterator
        evaluation_loader = iter(evaluation_loader)

        # If the number to save is none then we want to sa
        if(number_to_save is None):
            number_to_save = len(sequences_to_save)

        # Render!   
        number_saved = 0
        for i in tqdm(range(len(evaluation_loader)), desc="Saving Input and Output for models"):

            # Get the data
            data = next(evaluation_loader)

            # See if this is a sequence we want to save
            if(sequences_to_save is not None):
                if(i not in sequences_to_save):
                    print("Skipping", i)
                    continue

            # We saved all we needed to save
            if(number_saved > number_to_save):
                break
            number_saved += 1


            # Add that this is an evaluation stage
            data["stage"] = "evaluation"

            # Add in the model control parameters
            data["model_control_parameters"] = self.model_control_parameters

            # Do the forward pass over the data and get the model output
            outputs = self.do_forward_pass(data)

            # Move the data and outputs  to the CPU
            data = move_data_to_device(data, "cpu")
            outputs = move_data_to_device(outputs, "cpu")

            # Pack the dict
            save_dict = dict()
            save_dict["data"] = data
            save_dict["outputs"] = outputs

            # Save it
            save_file = "{}/seq_{:06d}.ptp".format(save_dir, i)
            torch.save(save_dict, save_file)

            # Compress it and delete it to save space
            compress_file(save_file, delete_file_after_compression=True)



    def _do_save_model_input_output_dynamics_model_propagation(self):

        ######################################################################################################################################################
        # Load all the configs
        ######################################################################################################################################################
        save_model_input_output_dynamics_model_propagation_config = get_mandatory_config("save_model_input_output_dynamics_model_propagation_config", self.qualitative_config, "qualitative_config")

        # If we should render anything at all
        do_save = get_mandatory_config("do_save", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config")

        # Get either the number to save or a list of which sequences to save
        number_to_save = get_optional_config_with_default("number_to_save", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config", default_value=None)
        sequences_to_save = get_optional_config_with_default("sequences_to_save", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config", default_value=None)
        assert((number_to_save is not None) or (sequences_to_save is not None))

        # If there is nothing to render then do nothing
        # No need to load any more configs in this case
        if((do_save == False) or (number_to_save <= 0)):
            return
        elif((number_to_save is not None) and (number_to_save == 0)):
            return
        elif((sequences_to_save is not None) and (len(sequences_to_save) == 0)):
            return

        # Get the name of the dynamics model and make sure its a real model
        dynamics_model_name = get_mandatory_config("dynamics_model_name", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config")
        assert(dynamics_model_name in self.base_model.get_submodels())

        # The number of samples to draw from the dynamics model
        number_of_samples = get_mandatory_config("number_of_samples", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config")
        assert(number_of_samples > 0)

        # The number of initial samples to use 
        number_of_initial_positions = get_mandatory_config("number_of_initial_positions", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config")
        assert(number_of_initial_positions > 0)

        # We need this condition to be met so that we can create the initial set of particles
        assert((number_of_samples % number_of_initial_positions) == 0)

        # The configs of the KDE that will be used to generate the initial values
        kde_samples_config = get_mandatory_config("kde_samples_config", save_model_input_output_dynamics_model_propagation_config, "save_model_input_output_dynamics_model_propagation_config")

        # Log that we are doing this
        self.logger.log("\n")
        self.logger.log("Saving model input and out for Dynamics Model Propagation")

        # Make a save directory 
        save_dir = "{}/dynamics_model_propagation_input_output/".format(self.qualitative_save_dir)
        ensure_directory_exists(save_dir)

        # create the dataloader
        batch_sizes = {"evaluation":1}
        evaluation_loader = self._create_data_loaders(batch_sizes, self.num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")


        # Need to make the dataloader an iterator
        evaluation_loader = iter(evaluation_loader)

        # put the models into evaluation mode
        for model_name in self.all_models.keys():
            self.all_models[model_name].eval()
        self.model.eval()

        # Get the dynamics model from the base model so we can use it alone
        dynamics_model = self.base_model.get_submodels()[dynamics_model_name]

        # If the number to save is none then we want to sa
        if(number_to_save is None):
            number_to_save = len(sequences_to_save)

        # Render!   
        number_saved = 0
        for i in tqdm(range(len(evaluation_loader)), desc="Dynamics Model Prop. Saving"):

            # Get the data
            data = next(evaluation_loader)

            # See if this is a sequence we want to save
            if(sequences_to_save is not None):
                if(i not in sequences_to_save):
                    print("Skipping", i)
                    continue

            # We saved all we needed to save
            if(number_saved > number_to_save):
                break
            number_saved += 1

            # Unpack the data
            xy_gt = data["xy_position_global_frame"]
            yaw_gt = data["roll_pitch_yaw"][..., -1].unsqueeze(-1)
            # actions = data.get("actions", None)

            # Move everything to the correct device
            xy_gt = xy_gt.to(self.device)
            yaw_gt = yaw_gt.to(self.device)
            # actions = actions.to(self.device)

            # Pack the xy and yaw into a single state
            state_gt = torch.cat([xy_gt, yaw_gt], dim=-1)


            # Remove the batch dim so we can put the sequence dim in the batch dim so we
            # can process the whole sequence at once
            state_gt = state_gt.squeeze(0)
            # actions2 = actions.squeeze(0)

            
            # print(actions.shape)
            # exit()

            # Create all the samples
            all_samples = []
            for r_idx in range(number_of_samples//number_of_initial_positions):

                # Create some actions
                actions = torch.zeros_like(state_gt)
                actions[:-1, ...] = state_gt[1:, :] - state_gt[:-1, :]
                rand = (torch.rand(actions.shape) * 2.0) - 1.0
                rand[:, 0:2] *= (self.evaluation_dataset.action_noise_xy * self.evaluation_dataset.pixels_per_meter)
                rand[:, 2] *= np.deg2rad(self.evaluation_dataset.action_noise_yaw_degrees)
                actions += rand.to(actions.device)
                actions[-1, ...] = 0
                actions = actions.float()

                # # Create the initial points kde
                # bandwidths = torch.tile(torch.FloatTensor(kde_samples_config["bandwidths"]).unsqueeze(0),[state_gt.shape[0], 1]).to(state_gt.device)
                # weights = torch.ones([state_gt.shape[0], 1], device=state_gt.device)
                # kde = KDE(kde_samples_config["distribution_types"], state_gt.unsqueeze(1), weights, bandwidths, particle_resampling_method="stratified")

                # Get the initial points
                # initial_points = kde.sample((number_of_initial_positions, ))

                # print(initial_points.shape)
                # print(actions.shape)

                # exit()

                # Make the samples
                initial_points = torch.tile(state_gt.unsqueeze(1),[1, number_of_initial_positions, 1])

                # run the dynamics model
                samples = dynamics_model(initial_points, actions)
                all_samples.append(samples)

            # Stack them
            all_samples = torch.cat(all_samples, dim=1)
            print(all_samples.shape)

            # Pack the dict
            save_dict = dict()
            save_dict["data"] = data
            save_dict["samples"] = all_samples

            # Save it
            save_file = "{}/seq_{:06d}.ptp".format(save_dir, i)
            torch.save(save_dict, save_file)

            # Compress it and delete it to save space
            compress_file(save_file, delete_file_after_compression=True)


    def _get_base_model(self, model):

        # Get the base model
        if(isinstance(model, torch.nn.DataParallel)):
            base_model = model.module
        else:
            base_model = model

        return base_model


    def _remove_batch_dim(self, data):

        # If its a string then we just do nothing
        if(isinstance(data, str)):
            return data

        # If its a tensor then remove the batch dim but make sure 
        # that the batch dim does indeed only have a size of 1
        elif(torch.is_tensor(data)):
            assert(data.shape[0] == 1)
            return data.squeeze(0)
        
        # If it is a dict then remove the batch dim for each entry in the dict
        elif(isinstance(data, dict)):
            new_data = dict()
            for k in data.keys():
                new_data[k] = self._remove_batch_dim(data[k])
            return new_data

        # If it is a list then remove the batch dim for each value in the list
        elif(isinstance(data, list)):            
            return [self._remove_batch_dim(v) for v in data]

        # If it is an int then there is nothing to do
        elif(isinstance(data, int)):            
            return data

        # If it is an long then there is nothing to do
        elif(isinstance(data, long)):            
            return data

        # If it is an float then there is nothing to do
        elif(isinstance(data, float)):            
            return data

        # This usually means its numpy
        elif(data is not None):
            assert(data.shape[0] == 1)
            return data.squeeze(0)

        # If the data is anything else then just return it
        # It isnt something that we can remove the batch dim from
        else:
            return data


    def _move_to_numpy(self, x):
        if(torch.is_tensor(x)):
            return x.detach().cpu().numpy()

        return x


    def _get_data_for_seq_index(self, x, seq_idx):

        if(x is None):
            return None

        return x[seq_idx]


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


    def _render_state_arrow(self, ax, x, y, yaw, color, yaw_offset=0,label=None, big_or_small="small"):

        # Move things to numpy if they are not
        x = self._move_to_numpy(x)
        y = self._move_to_numpy(y)
        yaw = self._move_to_numpy(yaw)

        # Apply a yaw offset sometimes
        yaw = yaw + yaw_offset

        assert(big_or_small in ["big", "small"])
        if(big_or_small == "big"):
            # width = 0.1
            # scale = 1.0
            width = 0.015
            scale = 10.0
        else:
            width = None
            scale = None

        # Render the arrow
        # ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, scale=scale, label=label)
        ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color, label=label, units="width", width=width, scale=scale)

        # Add in the circle
        # ax.add_patch(plt.Circle((x, y), 0.1, color=color))


    def _render_particles(self, ax, particles, particle_weights, color):
        
        # Move things to numpy if they are not
        particles = self._move_to_numpy(particles)

        ax.scatter(particles[..., 0], particles[..., 1], color=color, s=1)


    def _render_mean_particle_arrow(self, ax, particles, particle_weights, color):

        # Make sure we have all the data we need. If we dont then do nothing
        if((particles is None) or (particle_weights is None)):
            return 

        # Convert into a different representation
        transformed_particles = torch.zeros((particles.shape[0], 4), device=particles.device)
        transformed_particles[..., 0] = particles[..., 0]
        transformed_particles[..., 1] = particles[..., 1]
        transformed_particles[..., 2] = torch.sin(particles[..., 2])
        transformed_particles[..., 3] = torch.cos(particles[..., 2])

        # Compute the mean
        mean_transformed_particle  = torch.sum(transformed_particles * particle_weights.unsqueeze(-1), dim=0)

        # Transform back into the mean particle
        mean_particles = torch.zeros((3, ), device=particles.device)
        mean_particles[0] = mean_transformed_particle[0]
        mean_particles[1] = mean_transformed_particle[1]
        mean_particles[2] = torch.atan2(mean_transformed_particle[2], mean_transformed_particle[3])

        # Convert it to numpy
        mean_particles = self._move_to_numpy(mean_particles)

        # Extract the parts
        x = mean_particles[0]
        y = mean_particles[1]
        yaw = mean_particles[2]

        # Render the arrow
        ax.quiver(x, y, np.cos(yaw), np.sin(yaw), color=color)


    def _render_kde_posterior_angle(self, ax, particles, particle_weights, bandwidths, gt_yaw, rendering_configs):

        # Get the configs we need for this
        local_rendering_configs = rendering_configs["angle_kde_posterior_rendering"]
        do_render = local_rendering_configs["do_render"]
        angle_kde_posterior_particle_dims_to_use = local_rendering_configs["angle_kde_posterior_particle_dims_to_use"] 
        angle_kde_posterior_distribution_types = local_rendering_configs["angle_kde_posterior_distribution_types"] 


        # Make sure we actually want to render this
        if(do_render == False):
            return 

        # Make sure we have all the data we need. If we dont then do nothing
        if((particles is None) or (particle_weights is None) or (bandwidths is None)):
            return 

        # Extract the particle and bandwidth dims we want to use since we 
        # may not want to use the whole particle state for the loss (aka unsupervised latent dims)
        extracted_particles = [particles[..., pdim].unsqueeze(-1) for pdim in angle_kde_posterior_particle_dims_to_use]
        extracted_bandwidths = [bandwidths[..., pdim].unsqueeze(-1) for pdim in angle_kde_posterior_particle_dims_to_use]
        extracted_particles = torch.cat(extracted_particles, dim=-1)
        extracted_bandwidths = torch.cat(extracted_bandwidths, dim=-1)

        # Construct the KDE
        kde = KDE(angle_kde_posterior_distribution_types, extracted_particles.unsqueeze(0), particle_weights.unsqueeze(0), extracted_bandwidths.unsqueeze(0), particle_resampling_method="multinomial")

        # Compute the angles to evaluate at
        probe_points = torch.linspace(0, 2*np.pi, 1000).unsqueeze(0).unsqueeze(-1).to(particles.device)

        # Compute the probs of each of the points
        log_probs = kde.log_prob(probe_points)
        probs = torch.exp(log_probs)   

        # Squeeze to make 1D
        probe_points = probe_points.squeeze()
        probs = probs.squeeze()

        # Move to numpy for plotting
        probe_points = probe_points.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        # Convert the probe points from radians to degrees so we can better understand them when they are plotted
        probe_points = np.rad2deg(probe_points)

        # Plot the KDE
        ax.plot(probe_points, probs)

        # Plot the GT yaw
        gt_yaw = gt_yaw.numpy()
        gt_yaw = gt_yaw % (2.0 * np.pi)
        gt_yaw = np.rad2deg(gt_yaw)
        ax.axvline(x=gt_yaw, color="red")


        # Set the xticks 
        ax.set_xticks(np.arange(0, 361, step=45))



