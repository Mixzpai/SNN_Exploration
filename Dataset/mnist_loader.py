#-----------------------------------------------------------#
# Imports
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils

#-----------------------------------------------------------#
# Define a class to load and preprocess the MNIST dataset
class mnist_loader:
    def __init__(self):
        # Initialize the class and load the dataset
        self.dataset = self.get_mnist_dataset()

    def get_mnist_dataset(self):
        # Define the size of the subset to return
        subset_size = 10

        # Define the transformation to apply to the images
        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        
        # Load the MNIST training dataset
        mnist_train = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        # Create a subset of the dataset
        mnist_subset = utils.data_subset(mnist_train, subset_size)

        # Return a DataLoader for the subset
        return DataLoader(mnist_subset, batch_size=subset_size, shuffle=True)
