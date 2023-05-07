import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
batch_size = 8192  # Fastest epoch time on MPS on Mac M1 by trial and error
epochs = 200
learning_rate = 0.0002
beta1 = 0.5

# Set up device
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch built with MPS: {torch.backends.mps.is_built()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
device = torch.device("cpu" if not torch.backends.mps.is_available() else "mps")
print(f'Using device: {device}')


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)


# Load the MNIST dataset
transform = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


# Create the generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Set up the optimizers and loss function
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
loss_fn = nn.BCELoss()

# Training loop
for epoch in tqdm(range(epochs)):
    for i, (real_data, _) in enumerate(train_loader):
        real_data = real_data.to(device)
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        optimizer_D.zero_grad()
        real_output = discriminator(real_data)
        real_loss = loss_fn(real_output, real_labels)

        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        fake_loss = loss_fn(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = loss_fn(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch: {epoch}, D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.axis("off")
    plt.show()


# Generate synthetic digits
generator.eval()
n_samples = 64
noise = torch.randn(n_samples, latent_dim, 1, 1).to(device)
generated_samples = generator(noise).detach().cpu()

# Display the gallery of synthetic digits
imshow(torchvision.utils.make_grid(generated_samples, padding=2, normalize=True, nrow=8))
