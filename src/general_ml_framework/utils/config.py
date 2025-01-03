
# Python Imports
import os

# Module imports 
import yaml

def get_mandatory_config(config_name, config_dict, config_dict_name):

    if(config_name not in config_dict):
        print("Could not find \"{}\" in \"{}\"".format(config_name, config_dict_name))
        assert(False)

    return config_dict[config_name]



def get_mandatory_config_as_type(config_name, config_dict, config_dict_name, dtype):

    if(config_name not in config_dict):
        print("Could not find \"{}\" in \"{}\"".format(config_name, config_dict_name))
        assert(False)

    value = config_dict[config_name]

    assert(isinstance(value, dtype))

    return value


def get_optional_config_as_type_with_default(config_name, config_dict, config_dict_name, dtype, default_value=None):

    if(config_name not in config_dict):
        return default_value

    value = config_dict[config_name]
    
    assert(isinstance(value, dtype))

    return value




def get_optional_config_with_default(config_name, config_dict, config_dict_name, default_value=None):

    if(config_name not in config_dict):
        return default_value
    return config_dict[config_name]


def ensure_directory_exists(directory):
    '''
        Makes sure a directory exists.  If it does not exist then the directory is created

        Parameters:
            directory: The directory that needs to exist

        Returns:
            None
    '''
    if(not os.path.exists(directory)):
        os.makedirs(directory)

