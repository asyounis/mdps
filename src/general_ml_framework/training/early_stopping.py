
# Project Imports
from ..utils.config import *

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, configs):

        # extract the configs
        #    patience: how many epochs to wait before stopping when loss is not improving
        #    min_delta: minimum difference between new loss and old loss for new loss to be considered as an improvement
        #    start_offset: minimum number of steps before early stopping is considered
        self.patience = get_mandatory_config("patience", configs, "configs")
        self.min_delta = get_mandatory_config("min_delta", configs, "configs")
        self.start_offset = get_mandatory_config("start_offset", configs, "configs")
        self.max_lr_change = get_mandatory_config("max_lr_change", configs, "configs")

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.lr_change_counter = 0

    def __call__(self, val_loss):

        if(self.start_offset > 0):
            self.start_offset -= 1
            return

        if(self.best_loss == None):
            self.best_loss = val_loss

        elif((self.best_loss - val_loss) > self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            
        elif ((self.best_loss - val_loss) < self.min_delta):
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if (self.counter >= self.patience):
                print('INFO: Early stopping')
                self.early_stop = True

    def lr_changed(self):
        self.lr_change_counter += 1
        if(self.lr_change_counter <= self.max_lr_change):
            self.counter = 0


    def do_stop(self):
        return self.early_stop


    def get_save_dict(self):
        '''
            Get the save dict so that we can use to load this object from a save

            Returns:
                The save dict
        ''' 

        save_dict = dict()
        save_dict["patience"] = self.patience
        save_dict["min_delta"] = self.min_delta
        save_dict["start_offset"] = self.start_offset
        save_dict["max_lr_change"] = self.max_lr_change
        save_dict["counter"] = self.counter
        save_dict["best_loss"] = self.best_loss
        save_dict["early_stop"] = self.early_stop
        save_dict["lr_change_counter"] = self.lr_change_counter
        return save_dict

    def load_from_dict(self, saved_dict):
        '''
            Load this object from a dict

            Parameters:
                saved_dict: The dict to load from
        ''' 

        self.patience = saved_dict["patience"]
        self.min_delta = saved_dict["min_delta"]
        self.start_offset = saved_dict["start_offset"]
        self.max_lr_change = saved_dict["max_lr_change"]
        self.counter = saved_dict["counter"]
        self.best_loss = saved_dict["best_loss"]
        self.early_stop = saved_dict["early_stop"]
        self.lr_change_counter = saved_dict["lr_change_counter"]
