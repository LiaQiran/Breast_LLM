import json
from torch.utils.data import DataLoader, Dataset
from open_clip import create_model_from_pretrained, get_tokenizer
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import os


# Define a custom dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_image_path, list_txt, tokenizer, preprocess):
        # Initialize image paths, texts, tokenizer, and preprocessing function
        self.image_path = list_image_path
        self.tokenizer = tokenizer
        self.preprocess = preprocess

        # Ensure all elements in list_txt are strings
        if not all(isinstance(txt, str) for txt in list_txt):
            raise ValueError("All elements in list_txt should be strings.")

        # Tokenize texts using the provided tokenizer
        context_length = 256
        try:
            # Tokenize the entire list at once
            self.title_tokens = self.tokenizer(list_txt, context_length=context_length)  # Returns a tensor directly
        except Exception as e:
            print(f"Error during tokenization: {e}")
            self.title_tokens = None  # Handle the case where tokenization fails

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # Preprocess the image
        image = self.preprocess(Image.open(self.image_path[idx]).convert('RGB'))

        # Get the tokenized output for the current index
        input_ids = self.title_tokens[idx]  # Directly use the tensor output

        return image, input_ids


def get_file_paths(directory_path):
    # List to store file paths
    file_paths = []

    # Iterate over all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Get the full file path
            full_path = os.path.join(root, file)
            # Normalize the path to use forward slashes
            file_paths.append(full_path.replace("\\", "/"))

    return file_paths


def get_label_column(csv_file_path, image_paths):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Ensure the file paths in the CSV match the image paths
    # Assuming the CSV has 'filename' as the first column and 'description' as the second
    image_filenames = [os.path.basename(path) for path in image_paths]

    # Create a dictionary to map image filenames to their descriptions
    label_dict = dict(zip(df['filename'], df['description']))

    # Map the descriptions to the image filenames in the correct order
    label_list = [label_dict[filename] for filename in image_filenames if filename in label_dict]

    return label_list


# Load the pretrained model and tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Move the model to the device
model = model.to(device)


# Freeze all layers except the last two transformer blocks and the projection layer
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two blocks of the Vision Transformer
for param in model.visual.trunk.blocks[-2:].parameters():
    param.requires_grad = True

# Unfreeze the projection layer in the head
for param in model.visual.head.proj.parameters():
    param.requires_grad = True

# Prepare the optimizer (for trainable parameters only)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
)

# Prepare file paths
directory_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/train/images'
image_paths = get_file_paths(directory_path)

csv_file_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/train_descriptions.csv'  # Replace this with the path to your CSV file
label_list = get_label_column(csv_file_path, image_paths)

# Create dataset and dataloader
dataset = ImageTitleDataset(image_paths, label_list, tokenizer, preprocess)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)


# Define contrastive loss function (InfoNCE)
def contrastive_loss(logits_per_image, logits_per_text):
    # Ground truth similarity is the identity matrix
    ground_truth = torch.arange(len(logits_per_image), dtype=torch.long, device=device)

    # Cross-entropy loss between the logits and ground truth labels
    loss_img = nn.CrossEntropyLoss()(logits_per_image, ground_truth)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, ground_truth)

    # Average the image-text and text-image losses
    return (loss_img + loss_txt) / 2


# Directory to save model checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize a variable to store the best loss
best_loss = float('inf')

# Initialize lists to store losses
epoch_losses = []

num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0  # Track the loss for this epoch
    pbar = tqdm(train_dataloader, total=len(train_dataloader))

    for batch in pbar:
        optimizer.zero_grad()

        # Get the images and input_ids from the batch
        images, input_ids = batch

        # Move data to the specified device
        images = images.to(device)
        input_ids = input_ids.to(device)

        # Forward pass through the model
        image_features, text_features, logit_scale = model(images, input_ids)

        # Calculate logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # Compute contrastive loss
        total_loss = contrastive_loss(logits_per_image, logits_per_text)

        # Backward pass
        total_loss.backward()

        # Update the optimizer
        optimizer.step()

        # Update loss for this epoch
        epoch_loss += total_loss.item()

        # Update progress bar with current loss
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")

        # Save the best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with loss {best_loss:.4f} at {checkpoint_path}")

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    epoch_losses.append(avg_epoch_loss)  # Store the average epoch loss

    print(f"Epoch {epoch + 1} completed, Avg Loss: {avg_epoch_loss:.4f}")

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

# Save the plot
loss_plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
plt.savefig(loss_plot_path)
print(f"Loss curve saved at {loss_plot_path}")

# Display the plot
plt.show()
