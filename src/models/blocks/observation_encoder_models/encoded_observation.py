# Python Imports

# Package Imports
import torch

# Ali Package Import

# Project Imports


class EncodedObservation:

	def __init__(self, encoded_observation):
		self.encoded_observation = encoded_observation

		# Extract the shape and the device
		if(torch.is_tensor(self.encoded_observation)):
			self.shape = self.encoded_observation.shape
			self.device = self.encoded_observation.device
		elif(isinstance(self.encoded_observation, tuple)):
			self.shape = self.encoded_observation[0].shape
			self.device = self.encoded_observation[0].device
		elif(isinstance(self.encoded_observation, list)):
			self.shape = self.encoded_observation[0].shape
			self.device = self.encoded_observation[0].device
		else:
			assert(False)




	def get(self):
		return self.encoded_observation

	def get_subsequence_index(self, i):

		# If we are a pytorch tensor then slicing is easy
		if(torch.is_tensor(self.encoded_observation)):
			return self.encoded_observation[:, i]	

		# If we are a tuple then we need to slice out one by one
		if(isinstance(self.encoded_observation, tuple)):
			sliced_eo = [eo[:, i] for eo in self.encoded_observation]
			return tuple(sliced_eo)

		# If we are a list then we need to slice out one by one
		if(isinstance(self.encoded_observation, list)):
			sliced_eo = [eo[:, i] for eo in self.encoded_observation]
			return sliced_eo


		# Dont know how do handle this so exit
		assert(False)