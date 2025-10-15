#-----------------------------------------------------------#
# Imports
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

#-----------------------------------------------------------#
# Animation Hub for visualizing spike data
class spikevis:
    def __init__(self, spike_data):
        self.spike_data = spike_data
        self.sample = spike_data[:, 0, 0] # Take the first sample in the batch
        self.sample2 = self.sample.reshape((100,-1)) # Reshape for raster plotting

    def animate_spike_sample(self, fps = 30):
        """
        Animate a sample of rate-coded spikes over time.

        This function visualizes the temporal evolution of a single spike sample
        using Matplotlib animation utilities. Each frame in the animation corresponds
        to one time step of spike activity, allowing visual inspection of the dynamic
        firing behavior in the rate-coded representation.

        Parameters
        ----------
        fps : int, optional
            Frames per second of the animation (default is 30).

        Returns
        -------
        None
            Displays an animated visualization of spike activity.

        Notes
        -----
        - The method creates a Matplotlib figure and axis, then passes the current
        spike sample (`self.sample`) to the `splt.animator` function for animation.
        - The `event_source.interval` property is adjusted based on the desired
        frame rate, computed as `1000 / fps` (milliseconds per frame).
        - The function blocks execution until the Matplotlib window is closed.

        Example
        -------
        >>> visualizer = SpikeVisualizer(sample_spikes)
        >>> visualizer.animate_spike_sample(fps=24)
        # Opens a Matplotlib window showing the spiking pattern over time.
        """
        fig, ax = plt.subplots()
        ax.set_title("Rate-coded spikes (one sample)")
        anim = splt.animator(self.sample,fig, ax)
        anim.event_source.interval = 1000 / fps
        plt.show()

    def sample_raster(self):
        """

        """
        fig = plt.figure(facecolor='white', figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(self.sample2, ax, s=1.5, c="black")
        plt.title("Raster plot of one sample")
        plt.xlabel("Time step")
        plt.ylabel("Neuron number")
        plt.show()

    def sample_raster_sneuron(self):
        """

        """
        idx = 210  # Change this index to visualize different neurons
        fig = plt.figure(facecolor='white', figsize=(10, 4))
        ax = fig.add_subplot(111)
        splt.raster(self.sample.reshape(100, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
        plt.title("Input neuron " + str(idx))
        plt.xlabel("Time step")
        plt.yticks([])
        plt.show()

    def latency_raster(self):
        """

        """
        fig = plt.figure(facecolor="white", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(self.sample.view(100, -1), ax, s=25, c="black")

        plt.title("Input Layer")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        plt.show()

    def animate_latency(self,fps=30):
        """

        """
        fig, ax = plt.subplots()
        ax.set_title("Latency-coded spikes (one sample)")
        anim = splt.animator(self.sample, fig, ax)
        anim.event_source.interval = 1000 / fps
        plt.show()