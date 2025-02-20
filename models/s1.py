# models/s1.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the token embeddings."""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class VisuomotorTransformer(nn.Module):
    """
    System 1 (S1): Transformer-based visuomotor controller.
    Processes high-frequency visual and state inputs along with the latent vector from S2
    to produce a sequence of continuous actions.
    """

    def __init__(
        self,
        embed_dim,
        num_layers,
        num_heads,
        action_dim,
        seq_length,
        state_dim=512,
        latent_dim=512,
    ):
        super(VisuomotorTransformer, self).__init__()
        self.seq_length = seq_length
        # Dummy convolutional backbone for visual feature extraction.
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project the robot state from dimension 512 to embed_dim (256).
        self.state_proj = nn.Linear(state_dim, embed_dim)

        # Positional encoding to inject sequence order.
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_length + 1)
        # Use batch_first=True to have transformer input as (B, seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Project latent vector from S2 (512) to embed_dim (256).
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Final regression head to output continuous actions.
        self.regressor = nn.Linear(embed_dim, action_dim)

    def forward(self, image, state, latent):
        """
        image: (B, 3, H, W) – high-frequency image input.
        state: (B, 512) – high-frequency robot state.
        latent: (B, 512) – latent vector from System 2.
        """
        B = image.size(0)
        # Extract visual features.
        vision_feat = self.vision_backbone(image)  # (B, embed_dim, 1, 1)
        vision_feat = vision_feat.view(B, -1)  # (B, embed_dim)

        # Project the state.
        state_feat = self.state_proj(state)  # (B, embed_dim)

        # Fuse vision and state into a single token.
        token = vision_feat + state_feat  # (B, embed_dim)

        # Replicate token to form a sequence (dummy temporal replication).
        seq_tokens = token.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # (B, seq_length, embed_dim)

        # Process latent from S2 and prepend it as an extra token.
        latent_token = self.latent_proj(latent).unsqueeze(1)  # (B, 1, embed_dim)
        seq_tokens = torch.cat(
            [latent_token, seq_tokens], dim=1
        )  # (B, seq_length+1, embed_dim)

        # Add positional encoding.
        seq_tokens = self.pos_encoder(seq_tokens)  # (B, seq_length+1, embed_dim)

        # Process with transformer encoder (batch_first=True, so no transpose needed).
        transformer_out = self.transformer_encoder(
            seq_tokens
        )  # (B, seq_length+1, embed_dim)

        # Remove the latent token (first token) from the sequence.
        transformer_out = transformer_out[:, 1:, :]  # (B, seq_length, embed_dim)

        # Predict the continuous action sequence.
        actions = self.regressor(transformer_out)  # (B, seq_length, action_dim)
        return actions
