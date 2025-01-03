
# Python Imports

# Package Imports
import torch

# Project Imports
from .utils.config import *


class ModelSaverLoader:
	def __init__(self, models_dict, save_dir, logger):
			
		# Save for later
		self.models_dict = models_dict
		self.save_dir = save_dir
		self.logger = logger

	def save_models(self, epoch, is_best):

		# Save the models
		model_save_dir = "{}/models/epoch_{:05d}/".format(self.save_dir, epoch)
		self._save_models_in_dir(model_save_dir)

		# Save the models if its the best
		if(is_best):
			model_save_dir = "{}/models/best/".format(self.save_dir, epoch)
			self._save_models_in_dir(model_save_dir)

	@staticmethod 
	def load_models(model, pretrained_model_configs, logger):

		# If we are to load the full model then load it but make sure no other models have been specified
		if("full_model" in pretrained_model_configs):
			assert(len(pretrained_model_configs) == 1)


			# Load the model
			load_file = pretrained_model_configs["full_model"]
			state_dict = torch.load(load_file, map_location="cpu")
			model.load_state_dict(state_dict)
			logger.log("Loading \"Full Model\" from")
			logger.log("\t {}".format(load_file))
			return

		# Not the full model so we are good!

		# Load the internal models
		models_not_loaded = []
		internal_models = model.get_submodels()
		for model_name in internal_models.keys():

			# Nothing to load
			if(model_name not in pretrained_model_configs):
				models_not_loaded.append(model_name)
				continue

			# Load the model
			load_file = pretrained_model_configs[model_name]
			state_dict = torch.load(load_file, map_location="cpu")
			internal_models[model_name].load_state_dict(state_dict)
			logger.log("Loading \"{}\"".format(model_name))
			logger.log("\t {}".format(load_file))

		# Say which models we did not load
		if(len(models_not_loaded) != 0):
			logger.log("\n")
			logger.log("Did not load save files for: ")
			for model_name in models_not_loaded:
				logger.log("\t - \"{}\"".format(model_name))
		else:
			logger.log("\n")
			logger.log("Save file loaded for all models")
			

		# Check if there are any models that we were asked to load that we didnt load
		models_not_loaded = []
		for model_name in pretrained_model_configs:
			if(model_name not in internal_models):
				models_not_loaded.append(model_name)


		# Say which models we did not load
		if(len(models_not_loaded) != 0):
			logger.log("\n")
			logger.log("Model files were specified for the following models, but those models dont actually exist: ")
			for model_name in models_not_loaded:
				logger.log("\t - \"{}\"".format(model_name))



	def _save_models_in_dir(self, directory):

		# Make sure the directory exists
		ensure_directory_exists(directory)

		# Save the models state dicts
		for model_name in self.models_dict.keys():

			# Create the model save file
			model_save_filepath = "{}/{}.pt".format(directory, model_name)

			# Save the state dict
			torch.save(self.models_dict[model_name].state_dict(), model_save_filepath)	


