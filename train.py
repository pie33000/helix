# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.helix import Helix
from datasets_rob.open_x_embodiment import OpenXEmbodimentDataset
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE


def train():
    # Initialize dataset and dataloader.
    dataset = OpenXEmbodimentDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Helix model.
    model = Helix()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the regression loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in dataloader:
            image, state, text, high_freq_image, high_freq_state, target_actions = batch
            # Move batch data to the appropriate device.
            image = image.to(device)
            state = state.to(device)
            text = text.to(device)
            high_freq_image = high_freq_image.to(device)
            high_freq_state = high_freq_state.to(device)
            target_actions = target_actions.to(device)

            optimizer.zero_grad()
            # Forward pass through Helix.
            predicted_actions = model(
                image, state, text, high_freq_image, high_freq_state
            )
            loss = criterion(predicted_actions, target_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Save the trained model.
    torch.save(model.state_dict(), "helix_model.pth")
    print("Training complete and model saved.")


if __name__ == "__main__":
    train()
