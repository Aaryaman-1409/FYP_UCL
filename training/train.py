import sys
import torch
from torch import optim
from dataloader.get_motion_data import get_motion_data
from models.diffusion import Diffusion
from models.nextnet import NextNet


def train(model, data_loader, optimizer, device):
    model.train()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the NextNet model (adjust parameters as needed)
    nextnet_model = NextNet().to(device)

    # Initialize the Diffusion model with NextNet model instance
    diffusion_model = Diffusion(nextnet_model).to(device)

    # Specify the optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.001)

    # Load your motion data - adjust as necessary for your dataloader
    filename = sys.argv[1] if len(sys.argv) > 1 else "/home/aaryaman/Developer/Truebone_Z-OO/Ant/__Attack.bvh"
    motion_data = get_motion_data(filename)
    print(motion_data.shape)
    data_loader = torch.utils.data.DataLoader(motion_data, batch_size=64, shuffle=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train(diffusion_model, data_loader, optimizer, device)


if __name__ == "__main__":
    main()
