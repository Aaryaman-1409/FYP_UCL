import sys
import torch
from torch import optim
from dataloader.get_motion_data import get_motion_data
from models.diffusion import Diffusion
from models.nextnet import NextNet

BATCH_SIZE = 64
MAX_ITER = 50_000


def train(model, batch, optimizer, device):
    model.train()
    for data in batch:
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        print(f'Loss: {loss}')
        loss.backward()
        optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your motion data - adjust as necessary for your dataloader
    filename = sys.argv[1] if len(sys.argv) > 1 else "/home/aaryaman/Developer/Truebone_Z-OO/Spider/__Jump.bvh"
    motion_data = get_motion_data(filename)
    num_frames = motion_data.shape[2]
    batch = motion_data.repeat((BATCH_SIZE, 1, 1, 1))

    # Initialize the NextNet model (adjust parameters as needed)
    nextnet_model = NextNet(in_channels=num_frames, out_channels=num_frames).to(device)

    # Initialize the Diffusion model with NextNet model instance
    diffusion_model = Diffusion(nextnet_model).to(device)

    # Specify the optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=0.001)

    # Training loop
    for i in range(MAX_ITER):
        train(diffusion_model, batch, optimizer, device)
        if i % 1000 == 0:
            print('Iteration:', i)


if __name__ == "__main__":
    main()
