# config.py
# Hyperparameters and configuration settings for Helix.

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Model hyperparameters
S2_LATENT_DIM = 512  # Latent vector dimension from System 2 (PaliGemma2)
S1_TRANSFORMER_EMBED_DIM = 256  # Embedding dimension for the transformer in System 1
S1_NUM_LAYERS = 4  # Number of transformer encoder layers (dummy setting)
S1_NUM_HEADS = 4  # Number of attention heads
ACTION_DIM = 35  # 35 Degrees-of-Freedom for robot control
SEQ_LENGTH = 200  # Predict a sequence of 200 actions (200Hz control)
