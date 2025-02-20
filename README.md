# Helix

Helix is a dual-system neural architecture that combines high-level vision-language understanding with precise visuomotor control for robotic applications.

## Architecture

Helix consists of two main systems:

### System 2 (S2)
- High-level vision-language model based on PaliGemma2
- Processes low-frequency inputs (7-9 Hz)
- Integrates visual information, robot state, and text commands
- Generates semantic latent representations

### System 1 (S1)
- Transformer-based visuomotor controller
- Processes high-frequency inputs (200 Hz)
- Generates continuous action sequences
- Conditioned on latent vectors from System 2

## Installation

    git clone https://github.com/yourusername/helix.git
    cd helix
    pip install torch

## Usage

To train the model:

    python train.py

## Configuration

Key hyperparameters can be modified in `config.py`:

- Batch size: 16
- Learning rate: 1e-4
- Number of epochs: 10
- Action dimensions: 35 (DoF)
- Sequence length: 200

## Dataset

The project uses the Open X-Embodiment dataset format. The current implementation includes a dummy dataset class that can be replaced with the actual dataset implementation.

## Model Components

1. **System 2 (S2)**
   - Vision-language model for high-level understanding
   - Processes images, state vectors, and text commands
   - Outputs latent vectors of dimension 512

2. **System 1 (S1)**
   - Visuomotor transformer with 4 layers and 4 attention heads
   - Processes high-frequency visual and state inputs
   - Generates action sequences conditioned on S2 latents

## License

This project is licensed under the MIT License. For more information, please see [LICENSE](LICENSE).

## Acknowledgments

This implementation is inspired by [Figure AI's Helix architecture](https://www.figure.ai/news/helix).


