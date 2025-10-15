#-----------------------------------------------------------#
# Imports
from snntorch import spikegen

#-----------------------------------------------------------#
# Spike Latency Coding Class
class spikelatency:
    def __init__(self, dataset, num_steps):
        self.num_steps = int(num_steps)
        self.data = iter(dataset)
        self.data_it, self.targets_it = next(self.data)
        self.tau = 5
        self.threshold = 0.01

    def get_latency_spikes(self):
        """
            Generate latency-coded spikes from input data.

            Latency coding encodes the intensity of each pixel as the timing of a spike.
            Brighter pixels produce earlier spikes, while darker pixels result in later
            spikes or no spikes at all. The `spikegen.latency` function generates these
            spike events based on the input intensities, number of time steps, and
            specified parameters like membrane time constant (`tau`) and firing threshold.

            Returns
            -------
            torch.Tensor
                Tensor of shape [num_steps, batch_size, channels, height, width]
                containing binary spike events over time.

            Notes
            -----
            • Each pixel value X_ij ∈ [0, 1] is transformed into a spike time T_ij,
            where higher intensities lead to earlier spike times. The relationship
            can be approximated as:

                T_ij ≈ (1 - X_ij) * num_steps
        """
        return spikegen.latency(self.data_it,self.num_steps,self.threshold,self.tau,clip=True,normalize=True,linear=True)