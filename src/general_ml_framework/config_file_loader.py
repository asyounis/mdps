# Python Imports
import copy 
import sys
import re

# Package Imports
import yaml

# Project Imports
from .utils.config import *

class ConfigFileLoader:
    def __init__(self, root_config_file):

        # The list of variables that are reserved for the frameworks internal use
        self.RESERVED_VARIABLES = ["<framework_var_run_number>"]


        # Load the root config file
        root_configs = self._load_yaml_file(root_config_file)

        # Load the variables
        self.variables = self._load_variables(root_configs)

        # Self resolve the root config for variable names
        root_configs = ConfigFileLoader.resolve_variables(self.variables, root_configs)

        # Load the the model architecture configs
        self.model_architecture_configs = self._load_model_architecture_configs(root_configs)

        # Load the experiments
        self.experiment_configs = self._load_experiment_configs(root_configs)

        # check for unresolved variables
        # We need to change the max recursion depth since we can go deeeeeeeep but make 
        # sure we change it back when we are done
        sys.setrecursionlimit(100000)
        self._check_for_unresolved_variables(self.model_architecture_configs)
        self._check_for_unresolved_variables(self.experiment_configs)
        sys.setrecursionlimit(1000)

        
        # # print(yaml.dump(self.model_architecture_configs, allow_unicode=True, default_flow_style=False))
        # print(yaml.dump(self.experiments, allow_unicode=True, default_flow_style=False))
        
    def get_model_architecture_configs(self):
        return self.model_architecture_configs

    def get_experiment_configs(self):
        return self.experiment_configs

    def resolve_variables(variables, target):
        target = copy.deepcopy(target)

        if(isinstance(target, list)):
            return [ConfigFileLoader.resolve_variables(variables, x) for x in target]

        elif(isinstance(target, dict)):
            for key in target.keys():
                target[key] = ConfigFileLoader.resolve_variables(variables, target[key])

            return target

        elif(isinstance(target, str)):

            for var_name in variables:

                # Get the variable value
                var_value = variables[var_name]

                # If it is the variable name then do a full replacement
                if(target == var_name):
                    target = copy.deepcopy(var_value)

                # Otherwise is is just in the target and so we need to do string replacements
                elif(var_name in target): 

                    # If it is a string then we can do string replacement otherwise we need to do something fancier
                    if(isinstance(var_value, str)):
                        target = target.replace(var_name, var_value)
                    else:

                        # We have to first convert to a string then make the replacement
                        var_value_str = str(var_value)
                        target = target.replace(var_name, var_value_str)

            return target

        else:
            return target


    def _load_variables(self, root_configs):

        # There are no variables and so return an empty dict
        if("variables" not in root_configs):
            return dict()

        # We have variables!
        variables = root_configs["variables"]

        # Check to make sure all variables are in the form "<variable_name>"
        for variable_name in variables.keys():
            assert(variable_name[0] == "<")
            assert(variable_name[-1] == ">")

        # Make sure none of the reserved variables are set
        for reserved_variable in self.RESERVED_VARIABLES:
            assert(reserved_variable not in variables)

        return variables


    def _load_model_architecture_configs(self, root_configs):

        # The model architecture configs
        model_architecture_configs = dict()

        # Get the list of model architecture files to load
        if("model_architecture_files" in root_configs):
            model_architecture_files = root_configs["model_architecture_files"]

            # Load all the config files in order and recursively update the model configs
            for model_config_file in model_architecture_files:

                # Load the config file
                cfg = self._load_yaml_file(model_config_file)

                # Update the model root_configs
                model_architecture_configs = self._update_dicts_with_new_dict(model_architecture_configs, cfg)

                # Resolve the variables
                model_architecture_configs = ConfigFileLoader.resolve_variables(self.variables, model_architecture_configs)

        # Load any overrides that we have
        if("model_architecture_overrides" in root_configs):

            # We have overrides
            model_architecture_overrides = root_configs["model_architecture_overrides"]

            # Update the dict
            model_architecture_configs = self._update_dicts_with_new_dict(model_architecture_configs, model_architecture_overrides)

            # Resolve the variables
            model_architecture_configs = ConfigFileLoader.resolve_variables(self.variables, model_architecture_configs)


        return model_architecture_configs


    def _load_experiment_configs(self, root_configs):

        # Templates for experiments
        experiment_templates = dict()

        # Load experiment templates form files if we have any
        if("experiment_templates_import" in root_configs):
            experiment_templates_import = root_configs["experiment_templates_import"]

            # Load all the config files in order and recursively update the model configs
            for model_config_file in experiment_templates_import:

                # Load the config file
                cfg = self._load_yaml_file(model_config_file)

                # Get the templates and add them
                templates = cfg["experiment_templates"]
                self._add_experiment_templates(experiment_templates, templates)

        # Load the templates form the main config 
        if("experiment_templates" in root_configs):

            # Get the templates and add them
            templates = root_configs["experiment_templates"]
            self._add_experiment_templates(experiment_templates, templates)

        # The model architecture configs
        experiments = list()

        # Get the list of experiment files to load
        if("experiments_import" in root_configs):
            experiments_import = root_configs["experiments_import"]

            # Load all the config files in order and recursively update the model configs
            for model_config_file in experiments_import:

                # Load the config file
                cfg = self._load_yaml_file(model_config_file)

                # Update the experiments
                experiments.extend(cfg["experiments"])

        # If we have additional experiments defined then we should add them in
        if("experiments" in root_configs):            
            experiments.extend(root_configs["experiments"])

        # Make sure that there are no experiments that have the same name
        used_names = set()
        for exp in experiments:
            
            # Make sure the experiment is formatted correctly
            assert(len(list(exp.keys())) == 1)

            # Extract the key
            key = list(exp.keys())[0]

            # Make sure its unique
            if(key in used_names):
                print("Experiment \"{}\" appears twice in experiment list".format(key))
                assert(False)

        # Load any overrides that we have
        if("experiments_overrides" in root_configs):

            # Get the index of each of the experiments so we can update them later
            exp_idices = dict()
            for i, exp in enumerate(experiments):
                exp_idices[list(exp.keys())[0]] = i

            # We have overrides
            experiments_overrides = root_configs["experiments_overrides"]
            for exp_override in experiments_overrides:

                # Get the key
                assert(len(list(exp_override.keys())) == 1)
                key = list(exp_override.keys())[0]
                    
                # Make sure the key is for a valid experiment
                if(key not in exp_idices):
                    print("Key Missing: \"{}\"".format(key))
                    assert(key in exp_idices)

                # Update
                exp_idx = exp_idices[key]
                exp_dict = experiments[exp_idx]
                exp_dict = self._update_dicts_with_new_dict(exp_dict, exp_override)
                experiments[exp_idx] = exp_dict

        # Resolve variables
        experiments = ConfigFileLoader.resolve_variables(self.variables, experiments)

        # Process the experiments
        processed_experiments = []
        for experiment in experiments:

            # Unpack the experiment configs
            experiment_name = list(experiment.keys())[0]
            experiment_top_level_configs = experiment[experiment_name]

            # These are the final experiment configs
            experiment = dict()

            # # If we have a parameters file we need to load then load it and add it to this experiment
            # if("common_experiments_config_file" in experiment):
                
            #     # Load and update
            #     common_experiments_config_file = experiment["common_experiments_config_file"]
            #     common_experiments_parameters = self._load_yaml_file(common_experiments_config_file)
            #     experiment = self._update_dicts_with_new_dict(common_experiments_parameters, experiment)

            #     # Resolve the variable names
            #     experiment = ConfigFileLoader.resolve_variables(self.variables, experiment)

            # If we have a template to use then we should use it
            if("templates_to_use" in experiment_top_level_configs):
                
                # Load the template to uses
                templates_to_use = experiment_top_level_configs["templates_to_use"]

                # Add each of the templates back to back
                for template_to_use in templates_to_use:

                    # Make sure the template to use is valid
                    if(template_to_use not in experiment_templates):
                        print("Template \"{}\" is not defined".format(template_to_use))
                        assert(False)

                    # Get the configs for the template to and update the experiment
                    template_configs = experiment_templates[template_to_use]
                    experiment = self._update_dicts_with_new_dict(experiment, template_configs)

                    # Resolve the variable names
                    experiment = ConfigFileLoader.resolve_variables(self.variables, experiment)

            if("dataset_configs_file" in experiment_top_level_configs):
                dataset_configs_file = experiment_top_level_configs["dataset_configs_file"]
            elif("dataset_configs_file" in experiment):
                dataset_configs_file = experiment["dataset_configs_file"]
            else:
                dataset_configs_file = None
            # If we have a dataset params file to load
            if(dataset_configs_file is not None):

                # Load and update
                dataset_params = self._load_yaml_file(dataset_configs_file)
                experiment = self._update_dicts_with_new_dict(experiment, dataset_params)

                print("Loading dataset config file for experiment \"{}\"".format(experiment_name))
                print("\t{}".format(dataset_configs_file))

                # Resolve the variable names
                experiment = ConfigFileLoader.resolve_variables(self.variables, experiment)


            # Finally override with any of the top level configs
            experiment = self._update_dicts_with_new_dict(experiment, experiment_top_level_configs)

            # Resolve the variable names for the last time
            experiment = ConfigFileLoader.resolve_variables(self.variables, experiment)

            # We are done!!
            processed_experiments.append({experiment_name: experiment, })

        return processed_experiments


    def _load_yaml_file(self, file_path):
        '''
            Load the YAML file into a python dict

            Parameters:
                file_path: The file path to the YAML file

            Returns:
                The loaded dictionary
        '''

        # Read and parse the config file
        with open(file_path) as file:

            # Load the whole file into a dictionary and return
            return yaml.load(file, Loader=yaml.FullLoader)
            

    def _update_dicts_with_new_dict(self, target, new_stuff):
        '''
            Recursively updates a dictionary with new data. 
            This is not an in place operation.

            Parameters:
                target: the dictionary to update
                new_stuff: the new stuff to add to the dict 

            Returns
                The updated dict
        '''

        # Make a copy so we dont edit the original dictionary
        target = copy.deepcopy(target)


        if(isinstance(target, list)):
            assert(False)

        elif(isinstance(target, dict)):

            # Update!
            for key in new_stuff.keys():
            
                # If the key was not in the target then add it
                if(key not in target):
                    target[key] = new_stuff[key]

                # It was in the target so instead we need to update it
                else:

                    # If it is a dict then recursively update it
                    if(isinstance(new_stuff[key], dict)):
                        assert(isinstance(target[key], dict))
                        target[key] = self._update_dicts_with_new_dict(target[key], new_stuff[key])
                    
                    # Otherwise manually update it
                    else:
                        target[key] = new_stuff[key]   
        else:
            print(type(target))
            assert(False)



        return target


    def _check_for_unresolved_variables(self, target):


        if(isinstance(target, dict)):
            for key, value in target.items():
                self._check_for_unresolved_variables(key)
                self._check_for_unresolved_variables(value)

        elif(isinstance(target, list)):
            for value in target:
                self._check_for_unresolved_variables(value)   

        elif(isinstance(target, str)):

            # Exclude any reserved variables by removing them and then replacing them after the check
            REPLACEMENT_STRING_LEFT = "_-_-_-_-_-_-_-_"
            REPLACEMENT_STRING_RIGHT = "=-=-=-=-=-=-=-="
            for reserved_variable in self.RESERVED_VARIABLES:
                replace_name = reserved_variable.replace("<", REPLACEMENT_STRING_LEFT)
                replace_name = replace_name.replace(">", REPLACEMENT_STRING_RIGHT)
                target = target.replace(reserved_variable, replace_name)

            # Do the check
            pattern = re.compile(r"<[A-Za-z0-9_]+>", re.IGNORECASE)
            match_pattern = pattern.match(target)
            if(match_pattern is not None):
                print("Unresolved variable \"{}\"".format(target))
                assert(False)

            # Put it back
            # Exclude any reserved variables by removing them and then replacing them after the check
            REPLACEMENT_STRING_LEFT = "_-_-_-_-_-_-_-_"
            REPLACEMENT_STRING_RIGHT = "=-=-=-=-=-=-=-="
            for reserved_variable in self.RESERVED_VARIABLES:
                replace_name = reserved_variable.replace("<", REPLACEMENT_STRING_LEFT)
                replace_name = replace_name.replace(">", REPLACEMENT_STRING_RIGHT)
                target = target.replace(replace_name, reserved_variable)



    def _add_experiment_templates(self, experiment_templates, templates):


        # Add all the templates
        for template in templates:

            # Get the name of the template and its configs
            template_name = list(template.keys())[0]
            template_configs = template[template_name]

            # Make sure that this template is unique
            if(template_name in experiment_templates):
                print("Template named \"{}\" defined multiple times", template_name)
                assert(False)

            # We dont allow this since this will cause recursion issues
            assert("templates_to_use" not in template_configs)

            # Add it to the templates
            experiment_templates[template_name] = template_configs

        return experiment_templates