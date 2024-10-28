# Import necessary libraries
import os
import pandas as pd
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import torch
from transformers import CLIPProcessor

# Load the pretrained model and tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Initialize processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define your benign and malignant prompts
benign_prompt = "The ultrasound image shows a benign breast tumor."
malignant_prompt = "The ultrasound image shows a malignant breast tumor."


# Function to process and classify images in a directory
def classify_images_in_directory(image_dir, output_csv):
    results = []

    # Loop through all images in the directory (PNG files only)
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # Process only PNG files
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)

            # Preprocess the image using the processor
            inputs = processor(images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device)  # Move image tensor to GPU

            # Prepare prompts using the open_clip tokenizer
            prompts = [benign_prompt, malignant_prompt]
            tokenized_prompts = tokenizer(prompts)

            # Extract input_ids and attention_mask directly from the tokenized output
            input_ids = tokenized_prompts.to(device)

            # Forward pass through the model
            with torch.no_grad():
                image_features = model.encode_image(inputs['pixel_values'])  # Encode the image
                text_features = model.encode_text(input_ids)  # Encode the prompts

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity scores between the image and the prompts
                logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Get the predicted class based on the similarity scores
            predicted_class_index = logits_per_image.argmax().item()
            predicted_class = 'Benign' if predicted_class_index == 0 else 'Malignant'

            print(f"The predicted class for {filename} is: {predicted_class}")

            # Append result (image filename and predicted class)
            results.append((filename, predicted_class))

            # Clear GPU cache after each prediction to avoid memory issues
            torch.cuda.empty_cache()

    # Save results to a CSV file
    df = pd.DataFrame(results, columns=['filename', 'type'])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Set the directory containing the images
image_directory = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/images'

# Output CSV file
output_csv = 'classification_results.csv'

# Run the classification and save results to CSV
classify_images_in_directory(image_directory, output_csv)
#%%
import pandas as pd

# Load the ground truth (test.csv) and predictions (classification_results.csv)
ground_truth = pd.read_csv('C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test.csv')
predictions = pd.read_csv('classification_results.csv')

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
