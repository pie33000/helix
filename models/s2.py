# models/s2.py
import torch
import torch.nn as nn


# Dummy wrapper for the open-source VLM "PaliGemma2"
class PaliGemma2(nn.Module):
    def __init__(self, latent_dim):
        super(PaliGemma2, self).__init__()
        # In a production system, this would load pre-trained weights.
        # Here we simulate feature extraction with a couple of convolutional layers.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, latent_dim)

    def forward(self, image, state):
        # image: (B, 3, H, W)
        # state: (B, latent_dim) – dummy assumption: state vector already matches latent dim
        x = self.conv(image)  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        latent = self.fc(x)  # (B, latent_dim)
        # For simplicity, we fuse the state information by a simple addition.
        latent = latent + state
        return latent


class System2(nn.Module):
    """
    System 2 (S2): High-level vision–language model.
    Uses PaliGemma2 as backbone for visual features and fuses in text commands.
    """

    def __init__(self, latent_dim):
        super(System2, self).__init__()
        self.backbone = PaliGemma2(latent_dim)
        # For language commands, a simple text encoder is used.
        # In production, this might be a transformer or similar text encoder.
        self.text_encoder = nn.Sequential(
            nn.Embedding(10000, latent_dim), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, image, state, text):
        """
        image: (B, 3, H, W)
        state: (B, latent_dim)
        text: (B, seq_len) – tokenized text commands
        """
        latent_vision = self.backbone(
            image, state
        )  # Extract latent from image and state.
        # Compute average embedding over the tokenized text.
        text_embeds = self.text_encoder(text).mean(dim=1)  # (B, latent_dim)
        # Fuse visual and language information.
        latent = latent_vision + text_embeds
        return latent
