

# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.training.base_trainer import BaseTrainer
from general_ml_framework.utils.config import *
from general_ml_framework.training.data_plotter import DataPlotter


# Project Imports
from utils.memory import move_data_to_device, get_device_from_dict
from utils.general import nll_loss_xyr
from metrics.create_metrics import create_metric
from models.mdpf import MDPF
from models.mdps import MDPS


class FullSequenceTrainer(BaseTrainer):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, dataset_create_fn, load_from_checkpoint):
        super(FullSequenceTrainer, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model,  dataset_create_fn, load_from_checkpoint)

        # Get the loss function configs and create the loss function
        loss_configs = get_mandatory_config("loss_configs", self.training_configs, "training_configs")
        self.loss_function = create_metric(loss_configs)

        # Get the truncated BPTT value
        self.truncated_bptt_modulo = get_optional_config_with_default("truncated_bptt_modulo", self.training_configs, "training_configs", None)


    def do_forward_pass(self, data):

        # Mode the data to the correct device
        data = move_data_to_device(data, self.device)

        # Unpack the data so we can get some info
        observations = data["observations"]

        # Get some info
        batch_size = observations.shape[0]

        # Add some info to the data dict
        data["truncated_bptt_modulo"] = self.truncated_bptt_modulo

        with torch.autocast(device_type="cuda", enabled=False):
            
            # Run the model and get the output
            output = self.model(data)

        # Compute the loss
        loss = self.loss_function(output, data)

        return loss, batch_size




    def create_project_specific_dataplotters(self):

        # The data plotters dict
        data_plotters = dict()

        # Get which model it is
        base_model = self._get_base_model(self.model)

        # If it is MDPF then we have some dataplotters to add
        if(isinstance(base_model, MDPF) or isinstance(base_model, MDPS)):

            # Get the bandwidth models
            bandwidth_models = base_model.get_bandwidth_models()

            for model_name in bandwidth_models.keys():

                # Get the bandwidth model
                bandwidth_model = bandwidth_models[model_name]

                # The bandwidths
                bandwidths = bandwidth_model(None)

               # Create a bandwidth save dir
                bandwidth_epoch_data_plotter_save_dir = "{}/bandwidths/epoch/".format(self.save_dir)
                bandwidth_iteration_data_plotter_save_dir = "{}/bandwidths/iteration/".format(self.save_dir)

                # Create the data plotters
                for d in range(bandwidths.shape[-1]):
                    data_plotters["{}_epoch_{:02d}".format(model_name, d)] = DataPlotter("Bandwidth", "Epoch", "Bandwidths", bandwidth_epoch_data_plotter_save_dir, "{}_epoch_{:02d}.png".format(model_name, d), plot_modulo=self.data_plotter_plot_modulo)
                    data_plotters["{}_iteration_{:02d}".format(model_name, d)] = DataPlotter("Bandwidth", "Iteration", "Bandwidths", bandwidth_iteration_data_plotter_save_dir, "{}_iteration_{:02d}.png".format(model_name, d),plot_modulo=self.data_plotter_plot_modulo)


        return data_plotters


    def project_specific_end_of_epoch_fn(self, epoch):

        # Get which model it is
        base_model = self._get_base_model(self.model)

        # If it is MDPF then we have some dataplotters to add
        if(isinstance(base_model, MDPF) or isinstance(base_model, MDPS)):

            # Get the bandwidth models
            bandwidth_models = base_model.get_bandwidth_models()

            for model_name in bandwidth_models.keys():

                # Get the bandwidth model
                bandwidth_model = bandwidth_models[model_name]

                # The bandwidths
                bandwidths = bandwidth_model(None)

               # Create a bandwidth save dir
                bandwidth_data_plotter_save_dir = "{}/bandwidths/".format(self.save_dir)

                # Create the data plotters
                for d in range(bandwidths.shape[-1]):
                    self.data_plotters["{}_epoch_{:02d}".format(model_name, d)].add_value(bandwidths[0, d].item())




    def project_specific_end_of_training_batch_fn(self, epoch, step):

        # Get which model it is
        base_model = self._get_base_model(self.model)

        # If it is MDPF then we have some dataplotters to add
        if(isinstance(base_model, MDPF) or isinstance(base_model, MDPS)):

            # Get the bandwidth models
            bandwidth_models = base_model.get_bandwidth_models()

            for model_name in bandwidth_models.keys():

                # Get the bandwidth model
                bandwidth_model = bandwidth_models[model_name]

                # The bandwidths
                bandwidths = bandwidth_model(None)

               # Create a bandwidth save dir
                bandwidth_data_plotter_save_dir = "{}/bandwidths/".format(self.save_dir)

                # Create the data plotters
                for d in range(bandwidths.shape[-1]):
                    self.data_plotters["{}_iteration_{:02d}".format(model_name, d)].add_value(bandwidths[0, d].item())






    def _get_base_model(self, model):

        # Get the base model
        if(isinstance(model, torch.nn.DataParallel)):
            base_model = model.module
        else:
            base_model = model

        return base_model

