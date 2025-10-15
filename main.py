#-----------------------------------------------------------#
# SNN Encoding and Visualization Exploration
#   >>> Refactored snnTorch tutorial 1 
# Author: Mikka James Allen
# State: (Work in progress)
# To Do: 
#   >> Explore Delta Modulation
#   >> Add further explanations and comments (& Edit)

#-----------------------------------------------------------#
# Imports
from Dataset.mnist_loader import mnist_loader
from Encoder.Rate_coding.spikerates import rate_coding
from Encoder.Latency_Coding.spikelatency import spikelatency
from Encoder.Visualisation.spikevis import spikevis

#-----------------------------------------------------------#
#Attribution
"""
This implementation refactors and extends concepts from the open-source library snnTorch.

Original framework:
    snnTorch — maintained by the UCSC Neuromorphic Computing Group
    Initially developed by Jason K. Eshraghian (Lu Group, University of Michigan)
    Additional contributors: Vincent Sun, Peng Zhou, Ridger Zhu, Alexander Henkes,
    Steven Abreu, Xinxin Wang, Sreyes Venkatesh, gekkom, and Emre Neftci.
    Source: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html

Note:
    This project rewrites core ideas from snnTorch into a modular, class-based
    structure for experimentation and educational purposes.
"""

#-----------------------------------------------------------#
# Main execution block
if __name__ == "__main__":
    from Dataset.mnist_loader import mnist_loader
    # Load MNIST DataLoader 
    #   -> subset of dataset (navigate to mnist_loader to modify)
    mnist_data = mnist_loader().dataset

    # Generate spikes
    rate_coder = rate_coding(mnist_data, num_steps=100)
    spikes_re = rate_coder.get_rate_coded_spikes()
    latency_coder = spikelatency(mnist_data, num_steps=100)
    spikes_le = latency_coder.get_latency_spikes()

    # Visualise testing 1
    visualizer = spikevis(spikes_re)
    visualizer.animate_spike_sample()
    visualizer.sample_raster()
    visualizer.sample_raster_sneuron()
    
    # Visualise testing 2        
    visualizer = spikevis(spikes_le)
    visualizer.animate_latency()
    visualizer.latency_raster()