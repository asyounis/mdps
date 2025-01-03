# Python Imports

# Module Imports
import torch

# General ML Framework Imports
from general_ml_framework.utils.config import *

# Project Imports
import metrics.discrete_nll
import metrics.kde_nll
import metrics.combined
import metrics.dimension_regularizer
import metrics.pass_through

def create_metric(metric_config):

    # Get the name of the metric, we will use that to pick which metric to use
    name = get_mandatory_config("name", metric_config, "metric_config")

    # Note we that we dont need to give the metric a user readable name when we create it here 
    # The user readable name is more for the evaluation  

    if(name == "DicreteNLLMetric"):
        return metrics.discrete_nll.DicreteNLLMetric(None, metric_config)

    elif(name == "KDENLLMetric"):
        return metrics.kde_nll.KDENLLMetric(None, metric_config)

    elif(name == "PassThroughMetric"):
        return metrics.pass_through.PassThroughMetric(None, metric_config)

    elif(name == "CombinedMetric"):
        return metrics.combined.CombinedMetric(None, metric_config)

    elif(name == "DimensionRegularizer"):
        return metrics.dimension_regularizer.DimensionRegularizer(None, metric_config)

    else:
        print("Unknown metric named: \"{}\"".format(name))
        assert(False)
