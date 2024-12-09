import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

# Set random seed for reproducibility
torch.manual_seed(42)
class Generator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim):
        super(Generator, self).__init__()
        
        self.text_embedding = nn.Linear(text_embedding_dim, 256)
        
        self.init_size = 64 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim + 256, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        text_embedding = self.text_embedding(text_embedding)
        gen_input = torch.cat((noise, text_embedding), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim):
        super(Discriminator, self).__init__()
        
        self.text_embedding = nn.Linear(text_embedding_dim, 256)

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        text_embedding = self.text_embedding(text_embedding)
        img_features = self.model(img)
        img_features = img_features.view(img_features.size(0), -1)
        features = torch.cat((img_features, text_embedding), -1)
        validity = self.classifier(features)
        return validity
def train_cgan(generator, discriminator, dataloader, num_epochs, device):
    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (imgs, text_embeddings) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            # Configure input
            real_imgs = imgs.to(device)
            text_embeddings = text_embeddings.to(device)

            # Create labels
            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and generate images
            z = torch.randn(batch_size, 100).to(device)
            gen_imgs = generator(z, text_embeddings)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs, text_embeddings), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs, text_embeddings), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), text_embeddings), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
# Hyperparameters
latent_dim = 100
text_embedding_dim = 300  # Depends on your text embedding method
image_size = 64
batch_size = 32
num_epochs = 200

# Initialize models
generator = Generator(latent_dim, text_embedding_dim).to(device)
discriminator = Discriminator(text_embedding_dim).to(device)

# Create your custom dataset and dataloader here
dataloader = DataLoader("custom_dataset/train", batch_size=batch_size, shuffle=True)

# Train the model
train_cgan(generator, discriminator, dataloader, num_epochs, device)
