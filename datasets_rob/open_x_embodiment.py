# datasets/open_x_embodiment.py
import torch
from torch.utils.data import Dataset
import random


class OpenXEmbodimentDataset(Dataset):
    """
    Dummy implementation of the Open X-Embodiment Dataset.
    In a real-world setup, this class would load images, robot state vectors, and tokenized text commands.
    """

    def __init__(
        self, num_samples=1000, image_size=(3, 128, 128), state_dim=512, text_seq_len=10
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.state_dim = state_dim
        self.text_seq_len = text_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy low-frequency image and high-frequency image.
        image = torch.randn(self.image_size)
        high_freq_image = torch.randn(self.image_size)
        # Dummy state vectors (assume state dimension equals latent dim for simplicity).
        state = torch.randn(self.state_dim)
        high_freq_state = torch.randn(self.state_dim)
        # Dummy tokenized text command.
        text = torch.randint(low=0, high=10000, size=(self.text_seq_len,))
        # Dummy target action sequence: (SEQ_LENGTH, ACTION_DIM)
        target_actions = torch.randn(200, 35)
        return image, state, text, high_freq_image, high_freq_state, target_actions
