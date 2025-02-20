# models/helix.py
import torch.nn as nn
from models.s2 import System2
from models.s1 import VisuomotorTransformer
from config import (
    S2_LATENT_DIM,
    S1_TRANSFORMER_EMBED_DIM,
    S1_NUM_LAYERS,
    S1_NUM_HEADS,
    ACTION_DIM,
    SEQ_LENGTH,
)


class Helix(nn.Module):
    """
    Helix integrates both System2 (VLM) and System1 (visuomotor controller).
    """

    def __init__(self):
        super(Helix, self).__init__()
        self.system2 = System2(latent_dim=S2_LATENT_DIM)
        self.system1 = VisuomotorTransformer(
            embed_dim=S1_TRANSFORMER_EMBED_DIM,
            num_layers=S1_NUM_LAYERS,
            num_heads=S1_NUM_HEADS,
            action_dim=ACTION_DIM,
            seq_length=SEQ_LENGTH,
        )

    def forward(self, image, state, text, high_freq_image, high_freq_state):
        """
        - image, state, text: Low-frequency inputs for System2 (e.g. 7–9 Hz).
        - high_freq_image, high_freq_state: High-frequency inputs for System1 (e.g. 200 Hz).
        """
        # Generate a high-level latent vector from System2.
        latent = self.system2(image, state, text)  # (B, S2_LATENT_DIM)

        # Generate the action sequence from System1 conditioned on the latent.
        actions = self.system1(
            high_freq_image, high_freq_state, latent
        )  # (B, SEQ_LENGTH, ACTION_DIM)
        return actions
