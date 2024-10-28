import os
import pandas as pd
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
from PIL import Image
import torch

# Initialize processor, model, and classifier
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()  # Load pretrained weights
clf = PromptClassifier(model, ensemble=True)
clf.cuda()  # Move the classifier to GPU

# Define your benign and malignant prompts
benign_prompt = "The ultrasound image shows a benign breast tumor."
malignant_prompt = "The ultrasound image shows a malignant breast tumor."

# Prepare the list of prompts
prompts = [benign_prompt, malignant_prompt]


# Function to process and classify images in a directory
def classify_images_in_directory(image_dir, output_csv):
    results = []

    # Loop through all images in the directory (PNG files only)
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):  # Process only PNG files

            image_path = image_dir+'/'+filename
            image = Image.open(image_path)
            # Process the image using the processor
            inputs = processor(images=image, return_tensors="pt")

            # Define your benign and malignant prompts
            benign_prompt = "The ultrasound image shows a benign breast tumor."
            malignant_prompt = "The ultrasound image shows a malignant breast tumor."

            # Prepare the list of prompts
            prompts = [benign_prompt, malignant_prompt]

            # Process the prompts with padding and truncation
            prompt_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,  # Pad prompts to the same length
                truncation=True  # Truncate long prompts
            )

            # Ensure batch dimension (add a batch dimension if not present)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].unsqueeze(0) if len(
                prompt_inputs['input_ids'].shape) == 1 else prompt_inputs['input_ids']
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].unsqueeze(0) if len(
                prompt_inputs['attention_mask'].shape) == 1 else prompt_inputs['attention_mask']

            # Prepare prompt inputs as a dictionary (class names and corresponding prompts)
            prompt_dict = {
                'Benign': {'input_ids': prompt_inputs['input_ids'][0],
                           'attention_mask': prompt_inputs['attention_mask'][0]},
                'Malignant': {'input_ids': prompt_inputs['input_ids'][1],
                              'attention_mask': prompt_inputs['attention_mask'][1]},
            }

            # Add the prompt dictionary to inputs
            inputs['prompt_inputs'] = prompt_dict

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs['pixel_values'] = inputs['pixel_values'].cuda()  # Move image tensor to GPU
                for k, v in inputs['prompt_inputs'].items():
                    inputs['prompt_inputs'][k]['input_ids'] = v['input_ids'].unsqueeze(
                        0).cuda()  # Ensure batch dimension and move to GPU
                    inputs['prompt_inputs'][k]['attention_mask'] = v['attention_mask'].unsqueeze(
                        0).cuda()  # Ensure batch dimension and move to GPU

            # Make classification
            output = clf(pixel_values=inputs['pixel_values'], prompt_inputs=inputs['prompt_inputs'])
            print(output)

            # Print the output (logits for each class)

            logits = output['logits']
            class_names = output['class_names']

            # Find the index of the highest logit
            predicted_class_index = logits.argmax().item()

            # Get the corresponding class name
            predicted_class = class_names[predicted_class_index]

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



#%% Import required libraries
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
from PIL import Image
import torch

# Initialize processor, model, and classifier
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()  # Load pretrained weights
clf = PromptClassifier(model, ensemble=True)
clf.cuda()  # Move the classifier to GPU
image_dir='C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/images'
for filename in os.listdir(image_dir):

    image_path = image_dir  +  '/'  +  filename
    image = Image.open(image_path)

# Load and process the input image
#image = Image.open('C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/images/case71_BUSIS.png')

# # Convert image to RGB if necessary
# if image.mode != 'RGB':
#     image = image.convert('RGB')

    # Process the image using the processor
    inputs = processor(images=image, return_tensors="pt")

    # Define your benign and malignant prompts
    benign_prompt = "The ultrasound image shows a benign breast tumor."
    malignant_prompt = "The ultrasound image shows a malignant breast tumor."

    # Prepare the list of prompts
    prompts = [benign_prompt, malignant_prompt]

    # Process the prompts with padding and truncation
    prompt_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,  # Pad prompts to the same length
        truncation=True  # Truncate long prompts
    )

    # Ensure batch dimension (add a batch dimension if not present)
    prompt_inputs['input_ids'] = prompt_inputs['input_ids'].unsqueeze(0) if len(prompt_inputs['input_ids'].shape) == 1 else prompt_inputs['input_ids']
    prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].unsqueeze(0) if len(prompt_inputs['attention_mask'].shape) == 1 else prompt_inputs['attention_mask']

    # Prepare prompt inputs as a dictionary (class names and corresponding prompts)
    prompt_dict = {
        'Benign': {'input_ids': prompt_inputs['input_ids'][0], 'attention_mask': prompt_inputs['attention_mask'][0]},
        'Malignant': {'input_ids': prompt_inputs['input_ids'][1], 'attention_mask': prompt_inputs['attention_mask'][1]},
    }

    # Add the prompt dictionary to inputs
    inputs['prompt_inputs'] = prompt_dict

    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs['pixel_values'] = inputs['pixel_values'].cuda()  # Move image tensor to GPU
        for k, v in inputs['prompt_inputs'].items():
            inputs['prompt_inputs'][k]['input_ids'] = v['input_ids'].unsqueeze(0).cuda()  # Ensure batch dimension and move to GPU
            inputs['prompt_inputs'][k]['attention_mask'] = v['attention_mask'].unsqueeze(0).cuda()  # Ensure batch dimension and move to GPU

    # Make classification
    output = clf(pixel_values=inputs['pixel_values'], prompt_inputs=inputs['prompt_inputs'])
    print(output)


    # Print the output (logits for each class)

    logits = output['logits']
    class_names = output['class_names']

    # Find the index of the highest logit
    predicted_class_index = logits.argmax().item()

    # Get the corresponding class name
    predicted_class = class_names[predicted_class_index]

    print(f"The predicted class is: {predicted_class}")
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

# # Optionally, save the comparison to a CSV file for review
# comparison.to_csv('comparison_results.csv', index=False)
#
# print("Comparison saved to 'comparison_results.csv'")

