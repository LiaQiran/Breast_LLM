import os
import open_clip
import torch
from PIL import Image
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Paths
image_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/train/images'
output_csv = 'classification_results_linear_probe.csv'
custom_checkpoint_path = 'C:/Users/Qiran/Downloads/MedCLIP/biomedclip_best_model_epoch_16(1).pt'
labels_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/train.csv'

# Initialize CLIP model and tokenizer
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
checkpoint = torch.load(custom_checkpoint_path,
                        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint, strict=False)

# Freeze CLIP model parameters
for param in model.parameters():
    param.requires_grad = False

# Move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# Define Linear Probe
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


linear_probe = LinearProbe(input_dim=model.visual.output_dim).to(device)


# Custom Dataset for Image Embeddings and Labels
class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess, labels_df):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.labels_df = labels_df
        self.label_map = {'Benign': 0, 'Malignant': 1}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        filename = self.labels_df.iloc[idx]['filename']
        label = self.labels_df.iloc[idx]['type']

        # Convert label to int if it's not a string
        if isinstance(label, (int, float)):
            label = int(label)
        else:
            label = self.label_map[label]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image_input = self.preprocess(image).to(device)

        with torch.no_grad():
            image_embedding = model.encode_image(image_input.unsqueeze(0)).squeeze()
        return image_embedding, label


# Load labels and split data
labels_df = pd.read_csv(labels_path)
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Create DataLoaders
train_dataset = ImageDataset(image_directory, preprocess, train_df)
val_dataset = ImageDataset(image_directory, preprocess, val_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)

# Training loop
# Training loop
epochs = 5
for epoch in range(epochs):
    linear_probe.train()
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass through the linear probe
        outputs = linear_probe(images)

        # Ensure labels are on the same device and are of type long
        labels = labels.to(device).long()

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation code remains the same
    linear_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = linear_probe(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {accuracy:.2f}%")

