import os
import open_clip
import torch
from PIL import Image
import pandas as pd

# Define paths
image_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/images'
output_csv = 'classification_results1.csv'
custom_checkpoint_path = 'C:/Users/Qiran/Downloads/MedCLIP/biomedclip_best_model_epoch_16(1).pt'

# Initialize CLIP model and tokenizer
# Initialize CLIP model and tokenizer, ignoring the third returned value
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load custom checkpoint
checkpoint = torch.load(custom_checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint, strict=False)  # Load the weights directly

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define prompts for benign and malignant
benign_prompt = "The ultrasound image shows a benign breast tumor."
malignant_prompt = "The ultrasound image shows a malignant breast tumor."
prompts = [benign_prompt, malignant_prompt]

# Function to process and classify images
def classify_images_in_directory(image_dir, output_csv):
    results = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)

            # Preprocess the image
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Tokenize prompts
            text_inputs = tokenizer(prompts).to(device)

            # Forward pass
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                # Calculate similarity scores
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze()

                # Determine the predicted class based on highest similarity score
                predicted_class_index = similarity.argmax().item()
                predicted_class = "Benign" if predicted_class_index == 0 else "Malignant"

                print(f"The predicted class for {filename} is: {predicted_class}")

                # Append result
                results.append((filename, predicted_class))

    # Save results to CSV
    df = pd.DataFrame(results, columns=['filename', 'type'])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Run the classification and save results to CSV
classify_images_in_directory(image_directory, output_csv)



#%%
import pandas as pd

# Load the ground truth (test.csv) and predictions (classification_results.csv)
ground_truth = pd.read_csv('C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test.csv')
predictions = pd.read_csv('classification_results1.csv')

# Map predicted class names to 0 and 1
# 'Benign' -> 0, 'Malignant' -> 1
predictions['Predicted Type'] = predictions['type'].map({'Benign': 0, 'Malignant': 1})

# Merge the ground truth with predictions based on the filename
comparison = pd.merge(ground_truth, predictions, on='filename')

# Calculate accuracy
comparison['Correct'] = comparison['type_x'] == comparison['Predicted Type']
accuracy = comparison['Correct'].mean()

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
