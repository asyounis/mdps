

# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.training.base_trainer import BaseTrainer
from general_ml_framework.utils.config import *

# Project Imports
from utils.memory import move_data_to_device, get_device_from_dict


class TraditionalFFBSDynamicsTrainer(BaseTrainer):
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, dataset_create_fn, load_from_checkpoint):
        super(TraditionalFFBSDynamicsTrainer, self).__init__(experiment_name, experiment_configs, save_dir, logger, device, model,  dataset_create_fn, load_from_checkpoint)


        # Get the dynamics model
        self.dynamics_model = self.model.forward_dynamics_model



    def do_forward_pass(self, data):


        # Mode the data to the correct device
        data = move_data_to_device(data, self.device)

        # Unpack it
        xy_position_global_frame = data["xy_position_global_frame"]
        roll_pitch_yaw = data["roll_pitch_yaw"]
        assert("actions" not in data)

        # Create the states
        states = torch.cat([xy_position_global_frame, roll_pitch_yaw[..., -1].unsqueeze(-1)], dim=-1)

        # Get the x0 and x1
        x0 = states[:, :-1, ...]
        x1 = states[:, 1:, ...]

        # Flatten the batch dim
        x0 = torch.reshape(x0, [-1, x0.shape[-1]])
        x1 = torch.reshape(x1, [-1, x1.shape[-1]])

        # Run the dynamics model
        pred_x1 = self.dynamics_model(x0.unsqueeze(1), None)
        pred_x1 = pred_x1.squeeze(1)

        # For the x-y state do MSE
        pred_xy = pred_x1[..., :2]
        x1_xy = x1[..., 0:2]
        loss_xy = (pred_x1 - x1) ** 2
        loss_xy = torch.mean(loss_xy)

        # For the yaw state do MSE but make sure its close
        # https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        pred_yaw = pred_x1[..., 2]
        x1_yaw = x1[..., 2]
        diff = x1_yaw - pred_yaw
        diff = torch.atan2(torch.sin(diff), torch.cos(diff)) 
        loss_yaw = diff** 2
        loss_yaw = torch.mean(loss_yaw)

        # Final Loss
        loss = loss_xy + loss_yaw
        
        # Get the BS. This is a different batch size that specified in the config
        # because we dont care about the sequence length so we squash that into the batch size
        batch_size = x0.shape[0]

        return loss, batch_size


