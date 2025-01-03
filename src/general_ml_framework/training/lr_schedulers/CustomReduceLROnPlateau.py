
# Python Imports

# Package Imports
import torch

# Project Imports


class CustomReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(CustomReduceLROnPlateau, self).__init__(*args, **kwargs)


    def step(self, metrics, epoch=None):
        self.did_reduce_lr = False
        super(CustomReduceLROnPlateau, self).step(metrics, epoch=epoch)
        return self.did_reduce_lr

    def _reduce_lr(self, epoch):
        self.did_reduce_lr = True
        super(CustomReduceLROnPlateau, self)._reduce_lr(epoch=epoch)
