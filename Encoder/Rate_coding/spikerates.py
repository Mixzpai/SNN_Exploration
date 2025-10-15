#-----------------------------------------------------------#
# Imports
from snntorch import spikegen

# ----------------------------------------------------------- #
# Rate coding 
class rate_coding:
    """
        Spike encoding converts static MNIST pixel intensities into temporal spike trains, 
        allowing Spiking Neural Networks (SNNs) to process images as dynamic, event-based data.

        Each normalized pixel value X_ij ∈ [0,1] is treated as the probability of generating a spike 
        at any time step, modeled as a Bernoulli process:
                    
                R_ij ~ B(n=1, p=X_ij)
                where P(R_ij=1) = X_ij.

        - White pixels (X_ij ≈ 1) → high spike probability
        - Black pixels (X_ij ≈ 0) → no spikes

        This is known as *rate coding*: brighter pixels produce more frequent spikes over multiple time steps, 
        encoding information in firing rates rather than raw intensities.

        Spike encoding enables:
        - Biological realism (neurons fire over time)
        - Temporal learning (spatial + temporal pattern recognition)
        - Energy-efficient, event-driven computation on neuromorphic hardware

        In short, spike encoding bridges static MNIST images and the temporal domain required for 
        spiking computation.
    """
    def __init__(self, dataset, num_steps):
        self.num_steps = int(num_steps)
        self.data = iter(dataset)
        self.data_it, self.targets_it = next(self.data)
        self.gain = 0.25

    def get_rate_coded_spikes(self):
        """
            Generate rate-coded spikes from input data.

            The pixel intensity of each input is interpreted as the firing probability
            of a neuron at each time step. The `spikegen.rate` function generates binary
            spike events across time steps, where higher input intensities correspond
            to higher spike frequencies. An optional gain factor can scale the input
            values before spike generation.

            Returns
            -------
            torch.Tensor
                Tensor of shape [num_steps, batch_size, channels, height, width]
                containing binary spike events over time.

            Notes
            -----
            • Each pixel value X_ij ∈ [0, 1] represents the probability of firing
            at each timestep. Over many timesteps, the total spike count for that
            pixel approximates a Binomial process:

                total_spikes_ij ≈ Binomial(n = num_steps, p = X_ij)

            meaning the expected number of spikes is proportional to both the 
            pixel intensity and the number of timesteps:

                E[spikes_ij] = num_steps × X_ij

            • This models the principle of *rate coding*, where stronger stimuli 
            (brighter pixels) are represented by higher spike rates over time.

            Examples
            --------
            >>> inputs = torch.Tensor([0.0, 0.5, 1.0])
            >>> spikes = spikegen.rate(inputs, num_steps=3)
            >>> spikes.shape
            torch.Size([3, 3])  # 3 timesteps × 3 inputs

            >>> spikes
            tensor([[0., 1., 1.],
                    [0., 0., 1.],
                    [0., 1., 1.]])  # higher rate for stronger inputs
        """
        return spikegen.rate(self.data_it,self.num_steps,self.gain)