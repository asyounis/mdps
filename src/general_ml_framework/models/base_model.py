
# Package imports
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()


    def get_submodels(self):
        raise NotImplemented
