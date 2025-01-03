# Python Imports
import time

# Package Imports
import torch
import torch.distributions as D
import numpy as np

# Ali Package Import


def select_particles_multinomial_resampling_method(number_of_samples, weights):

    # There are no gradients when doing resampling so 
    # we might as well stop the gradients now to save computation time
    with torch.no_grad():

        # Create the distribution
        dist = D.Categorical(probs=weights)

        # Draw the samples
        samples = dist.sample((number_of_samples,))
        samples = torch.permute(samples, [1, 0])

        return samples

def select_particles_residual_resampling_method(number_of_samples, weights):

    # Get some info
    device = weights.device

    # There are no gradients when doing resampling so 
    # we might as well stop the gradients now to save computation time
    with torch.no_grad():

        # Get the number of particles we currently have
        current_number_of_particles = weights.shape[1]
        current_number_of_particles = float(current_number_of_particles)

        # Compute the number of copies needed aka the fraction of samples that need to come from each particle
        # This is fractional and not integers
        number_of_copies = number_of_samples * weights

        # Also compute the integer version of this which "deletes" the fractional part
        number_of_copies_int = torch.floor(number_of_copies)

        # Compute how many particles remain
        remaining_number_of_particles = number_of_samples - torch.sum(number_of_copies_int, dim=-1, keepdim=True)
        remaining_number_of_particles = remaining_number_of_particles.float()

        # Protect against dividing by zero
        # This is a hack to make the GPU implementation work
        # For remaining_number_of_particles<1, doing this hack effects the weights used
        # for multinomial resampling but we dont actually use samples from
        # from that resampling becayse there are no remaining_number_of_particles
        # So this is a moot point. But we need to do this since we still draw samples
        # to allow for parallelization
        remaining_number_of_particles[remaining_number_of_particles<1] = 1

        # Compute the weights to be used in the multinomial
        multinomial_weights = (current_number_of_particles * weights) - torch.floor(current_number_of_particles * weights)
        # multinomial_weights = multinomial_weights / remaining_number_of_particles.unsqueeze(-1)

        # This is a hack to make this run on the GPU
        # Basically what we are doing is making sure that the multinomial_weights is never a vector of 0
        # When it is a vector of 0 that means that remaining_number_of_particles==0 which means that we dont
        # need to get more samples from a uniform distribution (aka we have all the samples from just the)
        # integer counts part above.  Now for GPU implementation we still have to "draw samples" for the 
        # Cat dist defined by the vector of 0s which causes issues in pytorch due to normalization.
        # So the hack is that if the vector sums to 0 then just set it to some random value and normalize
        # This is fine since we wont actually use samples from this vector of 0s but this does prevent
        # issues when doing normalization.
        multinomial_weights_sum = torch.sum(multinomial_weights, dim=-1)            
        multinomial_weights_addition_to_prevent_nan = torch.ones_like(multinomial_weights_sum)
        multinomial_weights_addition_to_prevent_nan[multinomial_weights_sum > 1e-8] = 0
        multinomial_weights = multinomial_weights + multinomial_weights_addition_to_prevent_nan.unsqueeze(-1)
        multinomial_weights = torch.nn.functional.normalize(multinomial_weights, p=1.0, dim=-1, eps=1e-12, out=None)
        
        # Draw the remaining samples
        number_of_particles_to_sample = torch.max(remaining_number_of_particles).item() + 1
        dist = D.Categorical(probs=multinomial_weights)
        samples = dist.sample((int(number_of_particles_to_sample),))
        samples = torch.permute(samples, [1, 0])

        # Convert to a tensor of indices
        selected_indices = torch.zeros((weights.shape[0], number_of_samples), device=device)
        for b in range(weights.shape[0]):
            
            # Keep track of the tail
            idx = 0

            # Make copies of the particles
            for i in range(number_of_copies_int.shape[1]):
                c = int(number_of_copies_int[b, i].item())
                selected_indices[b, idx:idx+c] = i
                idx += c

            # Add in the random ones
            diff = selected_indices.shape[1] - idx
            selected_indices[b, idx:] = samples[b, 0:diff]

        return selected_indices.long()




def select_particles_stratified_resampling_method(number_of_samples, weights):

    # Get some info
    device = weights.device

    # There are no gradients when doing resampling so 
    # we might as well stop the gradients now to save computation time
    with torch.no_grad():


        # Create the stratified values over (0, 1]
        step_size = 1.0 / float(number_of_samples)
        stratified_samples = torch.arange(0, 1, step_size, device=device)
        stratified_samples = stratified_samples + (torch.rand(stratified_samples.shape, device=device) * step_size)
        
        # Compute the weights cumsum
        weights_cumsum = torch.cumsum(weights, dim=-1)

        # Index into the cumsum to extract the particle indices
        diff = stratified_samples.unsqueeze(-1) - weights_cumsum.unsqueeze(1)
        diff[diff > 0] = -np.inf
        indices = torch.argmax(diff, dim=-1)

        ##
        ## Edit:  Just get rid of this if we dont need it.  It doesnt hurt
        ##
        # # Shuffle the order of the indices to make sure that there is no ordering added to 
        # # the particle set because of stratified resampling
        # # This is needed because the above method of stratified resampling imposes order on the samples.
        # # The ordering comes from needing to use GPU friendly operations.
        # # Note: This is only needed if the input particles are not random ordered
        # # If they are then this step is not needed but it does not hurt so we keep it
        # shuffle_indices = torch.argsort(torch.rand(indices.shape, device=indices.device), dim=-1)
        # indices = torch.gather(indices, dim=-1, index=shuffle_indices)

        return indices