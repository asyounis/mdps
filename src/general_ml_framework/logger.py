# Python Imports
import os
import shutil

# Project Imports
from .utils.config import *

class Logger:
	def __init__(self, save_dir):

		# Save some important info	
		self.save_dir = save_dir

		# Make sure the directory exists
		ensure_directory_exists(self.save_dir)

		# create the output file
		self.output_file = "{}/output.txt".format(self.save_dir)

		# If the output file exists then move it
		if(os.path.isfile(self.output_file)):
			old_output_file = "{}/output_old.txt".format(self.save_dir)
			shutil.move(self.output_file, old_output_file)

	def log(self, text, print_to_terminal=True):

		# Make sure the text is a string. If it isnt then convert it to a string
		if(isinstance(text, str) == False):
			text = str(text)

		# Print the text if we are verbose
		if(print_to_terminal):
			print(text)

		# Save the text to the file
		with open(self.output_file, "a") as f:
			f.write(text)
			f.write("\n")


	def log_warning(self, text, print_to_terminal=True):

		self.log("", print_to_terminal=print_to_terminal)
		self.log("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING", print_to_terminal=print_to_terminal)
		self.log(text, print_to_terminal=print_to_terminal)
		self.log("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING", print_to_terminal=print_to_terminal)
		self.log("", print_to_terminal=print_to_terminal)

	def log_error(self, text, print_to_terminal=True):

		self.log("", print_to_terminal=print_to_terminal)
		self.log("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR", print_to_terminal=print_to_terminal)
		self.log(text, print_to_terminal=print_to_terminal)
		self.log("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR", print_to_terminal=print_to_terminal)
		self.log("", print_to_terminal=print_to_terminal)
