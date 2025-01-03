# Python Imports
import sys
import random

# Package Imports
import yaml
import torch 
import numpy as np 

# General ML Framework Imports
from general_ml_framework.experiment_runner import ExperimentRunner

# Project imports

# The Trainers 
from trainers.full_sequence_trainer import FullSequenceTrainer
from trainers.weight_model_trainer import WeightModelTrainer
from trainers.traditional_FFBS_dynamics_trainer import TraditionalFFBSDynamicsTrainer

# The Evaluators 
from evaluators.bearings_only_evaluator import BearingsOnlyEvaluator
from evaluators.kitti_evaluator import KittiEvaluator
from evaluators.mapillary_evaluator import MapillaryEvaluator
from evaluators.noisy_position_evaluator import NoisyPositionEvaluator



# The Models
from models.orienternet import OrienterNet
from models.mdpf import MDPF
from models.mdps import MDPS
from models.lstm_map_data import LSTMMapData
from models.embedding_maps_and_images import EmbeddingMapsAndImages
from models.beyond_cross_view_retrieval import BeyondCrossViewRetrieval
from models.traditional_FFBS import TraditionalFFBS

# The Datasets
from datasets.mapillary_dataset import MapillaryDataset
from datasets.mapillary_custom_splits_dataset import MapillaryCustomSplitsDataset
from datasets.bearings_only_dataset import BearingsOnlyDataset
from datasets.kitti_dataset import KittiDataset
from datasets.noisy_position_dataset import NoisyPositionDataset


# The metrics
# from metrics.translation_xy_mse import TranslationXYMSE
# from metrics.translation_xy_rmse import TranslationXYRMSE
# from metrics.rotation_mse import RotationMSE
# from metrics.rotation_rmse import RotationRMSE
from metrics.kde_nll import KDENLLMetric
from metrics.distance_recall import DistanceRecallMetric, MultipleDistanceRecallMetric
from metrics.distance_recall_xy_vehicle_frame import DistanceRecallXYVehicleFrameMetric, MultipleDistanceRecallXYVehicleFrameMetric

def main():

    # # Set the random seeds for this application to make things deterministic 
    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    # Make things faster 
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False 

    # Add some changes to pytorch for SPEEED
    torch.set_float32_matmul_precision("medium")

    # Create the experiment runner
    experiment_runner = ExperimentRunner()

    # Add model file things
    experiment_runner.add_model("orienternet", OrienterNet)
    experiment_runner.add_model("mdpf", MDPF)
    experiment_runner.add_model("mdps", MDPS)
    experiment_runner.add_model("lstm_map_data", LSTMMapData)
    experiment_runner.add_model("embedding_maps_and_images", EmbeddingMapsAndImages)
    experiment_runner.add_model("beyond_cross_view_retrieval", BeyondCrossViewRetrieval)
    experiment_runner.add_model("TraditionalFFBS", TraditionalFFBS)



    # Add Datasets
    experiment_runner.add_dataset("mapillary", MapillaryDataset)
    experiment_runner.add_dataset("mapillary_custom_split", MapillaryCustomSplitsDataset)
    experiment_runner.add_dataset("bearings_only", BearingsOnlyDataset)
    experiment_runner.add_dataset("kitti", KittiDataset)
    experiment_runner.add_dataset("noisy_position", NoisyPositionDataset)

    # Add Trainers
    experiment_runner.add_trainer("full_sequence_trainer", FullSequenceTrainer)
    experiment_runner.add_trainer("weight_model_trainer", WeightModelTrainer)
    experiment_runner.add_trainer("traditional_FFBS_dynamics_trainer", TraditionalFFBSDynamicsTrainer)




    # Add Evaluators
    experiment_runner.add_evaluator("bearings_only_evaluator", BearingsOnlyEvaluator)
    experiment_runner.add_evaluator("mapillary_evaluator", MapillaryEvaluator)
    experiment_runner.add_evaluator("kitti_evaluator", KittiEvaluator)
    experiment_runner.add_evaluator("noisy_position_evaluator", NoisyPositionEvaluator)

    # Add all the metrics
    # experiment_runner.add_metric("translation_xy_mse", TranslationXYMSE)    
    # experiment_runner.add_metric("translation_xy_rmse", TranslationXYRMSE)    
    # experiment_runner.add_metric("rotation_mse", RotationMSE)    
    # experiment_runner.add_metric("rotation_rmse", RotationRMSE)    
    experiment_runner.add_metric("kde_nll", KDENLLMetric)
    experiment_runner.add_metric("distance_recall", DistanceRecallMetric)
    experiment_runner.add_metric("multiple_distance_recall", MultipleDistanceRecallMetric)
    experiment_runner.add_metric("distance_recall_xy_vehicle_frame_metric", DistanceRecallXYVehicleFrameMetric)
    experiment_runner.add_metric("multiple_distance_recall_xy_vehicle_frame_metric", MultipleDistanceRecallXYVehicleFrameMetric)



    # Run!
    experiment_runner.run()

    print("All Done running everything! Exiting....")
    exit()

if __name__ == '__main__':
	main()