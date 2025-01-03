


# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
from utils.memory import move_data_to_device
from kde.get_class import get_probability_class
import metrics.create_metrics
from metrics.metrics_base import MetricsBase


class CombinedMetric(MetricsBase):
    def __init__(self, name, configs):
        super(CombinedMetric, self).__init__(name, configs)

        # All the metrics and their mixing cooefs
        self.metrics = []
        self.alphas = []
        self.metric_labels = []

        # Get all the metric configs
        all_metric_configs = get_mandatory_config("metric_configs", configs, "configs")
        for metric_configs in all_metric_configs:

            # Unpack the specific metric configs
            metric_label = list(metric_configs.keys())[0]
            metric_specific_configs = metric_configs[metric_label]

            # Create the metric
            metric = metrics.create_metrics.create_metric(metric_specific_configs)

            # Get the mixing cooef
            alpha = get_mandatory_config("alpha", metric_specific_configs, "metric_specific_configs")

            # Save for later
            self.metrics.append(metric)
            self.alphas.append(alpha)
            self.metric_labels.append(metric_label)


    def get_metric_value_for_current_data(self, model_output, data):

        # the total loss
        total_loss = 0

        for i in range(len(self.metrics)):

            # Get the metric and its alpha value
            metric = self.metrics[i]
            alpha = self.alphas[i]

            # Compute the loss
            loss = metric(model_output, data)

            # Add it to the total loss
            total_loss += (alpha * loss)

        return total_loss

    # This metric can only operate on single batches at a time and cannot report aggregated information
    def add_values(self, model_output, data):
        raise NotImplemented

    # This metric can only operate on single batches at a time and cannot report aggregated information
    def get_aggregated_result(self):
        raise NotImplemented