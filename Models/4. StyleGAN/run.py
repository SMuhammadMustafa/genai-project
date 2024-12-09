import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt


# Load CLIP model to encode text descriptions
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def get_text_embedding(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, output_channels=3):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        
        # Fully connected layer for text embeddings
        self.text_fc = nn.Linear(text_embedding_dim, latent_dim)
        
        # StyleGAN generator layers (simplified for this example)
        self.fc1 = nn.Linear(latent_dim * 2, 512)  # Latent + Text embedding
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_channels)

    def forward(self, z, text_embeddings):
        # Transform text embeddings to the same latent dimension
        text_features = self.text_fc(text_embeddings)
        
        # Concatenate the latent vector and text features
        combined_input = torch.cat((z, text_features), dim=1)
        
        # Pass through the network
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for image output
        return x
class ConditionalDiscriminator(nn.Module):
    def __init__(self, text_embedding_dim, input_channels=3):
        super(ConditionalDiscriminator, self).__init__()
        self.text_fc = nn.Linear(text_embedding_dim, 256)
        
        # Discriminator layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256 + 256, 1)  # Concatenate image features with text features

    def forward(self, img, text_embeddings):
        # Apply convolutional layers
        x = torch.relu(self.conv1(img))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get text features
        text_features = self.text_fc(text_embeddings)
        
        # Concatenate image features with text features
        x = torch.cat((x, text_features), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Initialize models
latent_dim = 128
text_embedding_dim = 512  # From CLIP
generator = ConditionalGenerator(latent_dim, text_embedding_dim)
discriminator = ConditionalDiscriminator(text_embedding_dim)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Adversarial loss
adversarial_loss = nn.BCELoss()

# Dummy dataset for illustration (you should replace this with your own dataset)
def generate_fake_data(batch_size):
    z = torch.randn(batch_size, latent_dim)
    return z

# Training Loop
epochs = 1000
for epoch in range(epochs):
    for i in range(100):  # Assume 100 iterations per epoch
        # Step 1: Train Discriminator
        real_images = torch.randn(8, 3, 64, 64)  # Replace with real images
        text = ["A red ball on the table"] * 8  # Example text batch
        real_labels = torch.ones(8, 1)
        fake_labels = torch.zeros(8, 1)
        
        # Get text embeddings
        text_embeddings = get_text_embedding(text)
        
        # Generate fake images
        z = generate_fake_data(8)
        fake_images = generator(z, text_embeddings)
        
        # Discriminator loss
        real_pred = discriminator(real_images, text_embeddings)
        fake_pred = discriminator(fake_images, text_embeddings)
        
        d_loss_real = adversarial_loss(real_pred, real_labels)
        d_loss_fake = adversarial_loss(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        # Step 2: Train Generator
        fake_images = generator(z, text_embeddings)
        fake_pred = discriminator(fake_images, text_embeddings)
        
        g_loss = adversarial_loss(fake_pred, real_labels)
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        
    # Print progress
    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
def generate_image_from_text(text):
    text_embedding = get_text_embedding([text])
    z = torch.randn(1, latent_dim)
    generated_image = generator(z, text_embedding)
    plt.imshow(generated_image.detach().numpy().transpose(1, 2, 0))
    plt.show()

# Example usage
generate_image_from_text("A red apple on a wooden table")
