# Python Imports

# Package Imports

# Ali Package Import

# Project Imports

class ProbabilityDistribution:

    def __init__(self):
        
        # By default the distributions are made up of just 1 distribution
        self.number_of_internal_distributions = 1

    def get_device(self):
        return NotImplemented

    def sample(self, shape):
        raise NotImplemented

    def log_prob(self, x, do_normalize_weights=True):
        raise NotImplemented

    def log_prob_separate(self, x, do_normalize_weights=True):

        # Compute the log prob and just pack it
        log_prob = self.log_prob(x, do_normalize_weights=do_normalize_weights)
        return [log_prob]