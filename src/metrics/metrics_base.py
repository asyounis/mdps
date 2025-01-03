# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from utils.memory import move_data_to_device
from kde.get_class import get_probability_class
import metrics.create_metrics


class MetricsBase:
    def __init__(self, name, configs):

        # Keep for later
        self.name = name
        self.configs = configs

        # The things we need to keep track of for this metric
        self.all_values = []

    def __call__(self, model_output, data):        
        return self.get_metric_value_for_current_data(model_output, data)


    def get_metric_value_for_current_data(self, model_output, data):

        # Clear whatever we currently have
        self.reset()

        # add the value
        self.add_values(model_output, data)

        # Get the mean
        mean = self.get_aggregated_result()[0]["mean"]

        # Clear the data again
        self.reset()

        return mean

    def reset(self):
        
        # Clear out the values
        self.all_values = []

    def add_values(self, model_output, data):
    	raise NotImplemented

    def get_aggregated_result(self):
    	raise NotImplemented