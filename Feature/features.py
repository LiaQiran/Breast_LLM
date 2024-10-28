import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import os
# Feature Extraction Functions

def compute_margin_features(contour):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    smoothness = np.std([cv2.pointPolygonTest(contour, (int(point[0][0]), int(point[0][1])), True) for point in approx])
    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1) % len(approx)][0]
        p3 = approx[(i+2) % len(approx)][0]
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    spikiness = np.sum(np.array(angles) < np.pi / 6)
    return smoothness, spikiness

def compute_shape_features(contour):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        aspect_ratio = major_axis / minor_axis
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if aspect_ratio < 1.2 and circularity > 0.75:
            shape = "Round"
        elif aspect_ratio < 1.5 and circularity > 0.6:
            shape = "Oval"
        else:
            shape = "Irregular"
        return shape, aspect_ratio, circularity
    else:
        return "Irregular", None, None

def compute_posterior_acoustic_features(contour, gray_image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y + h:y + 2 * h, x:x + w]
    if roi.size == 0:
        return "No Posterior Acoustic Features", None, None, roi
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)
    enhancement_threshold = 80
    shadowing_threshold = 60
    if mean_intensity > enhancement_threshold:
        posterior_feature = "Posterior Acoustic Enhancement"
    elif mean_intensity < shadowing_threshold:
        posterior_feature = "Posterior Acoustic Shadowing"
    else:
        posterior_feature = "No Posterior Acoustic Features"
    return posterior_feature, mean_intensity, std_intensity, roi

def compute_glcm_features(contour, gray_image):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y:y + h, x:x + w]
    if roi.size == 0:
        return "Complex", None, None, None, None
    glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    if energy > 0.3 and homogeneity > 0.3:
        echo_pattern = "Anechoic"
    elif contrast < 1000 and correlation > 0.5:
        echo_pattern = "Hypoechoic"
    elif contrast < 2000 and energy > 0.1:
        echo_pattern = "Isoechoic"
    elif contrast >= 2000 or correlation < 0.5:
        echo_pattern = "Hyperechoic"
    else:
        echo_pattern = "Complex"
    return echo_pattern, contrast, correlation, energy, homogeneity

def compute_lesion_boundary_feature(contour, gray_image, k=5):
    contour_points = contour.reshape(-1, 2)
    distance_map = np.zeros_like(gray_image, dtype=np.uint8)
    for point in contour_points:
        distance_map[point[1], point[0]] = 1
    distance_map = cv2.distanceTransform(1 - distance_map, cv2.DIST_L2, 5)
    surrounding_tissue = []
    outer_mass = []
    for y in range(gray_image.shape[0]):
        for x in range(gray_image.shape[1]):
            if distance_map[y, x] < k:
                surrounding_tissue.append(gray_image[y, x])
            elif distance_map[y, x] < 2 * k:
                outer_mass.append(gray_image[y, x])
    avg_tissue_intensity = np.mean(surrounding_tissue) if surrounding_tissue else 0
    avg_mass_intensity = np.mean(outer_mass) if outer_mass else 0
    lbd = avg_tissue_intensity - avg_mass_intensity # to get the ration difference
    return lbd


import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops


# Feature Extraction Functions
# (Keep all previous feature extraction functions unchanged)

# Function to generate description
def generate_description(result, cancer_type):
    if cancer_type == 0:
        description = "The image shows a benign breast cancer. "
    else:
        description = "The image shows a malignant breast cancer. "

    description += "The breast cancer area is "
    if result["lbd"] > 5:
        description += "with well-defined margins, appears brighter than the surrounding tissue "
    elif result["lbd"] < -5:
        description += "with well-defined margins, appears darker than the surrounding tissue "
    else:
        description += "with blurred margins."

    description += " and with " + result["posterior_feature"].lower() + '. '
    description += "The mass is located in "

    mass_x = result["centroid_x"]
    mass_y = result["centroid_y"]
    shape = result["size"]

    # Calculate the center of the mass
    mass_center_x = mass_x + shape[0] / 2
    mass_center_y = mass_y + shape[1] / 2

    if mass_center_x < shape[0] / 3:
        description += "left "
    elif mass_center_x > 2 * shape[1] / 3:
        description += "right "
    else:
        description += "center "

    if mass_center_y < shape[1] / 3:
        description += "top "
    elif mass_center_y > 2 * shape[1] / 3:
        description += "bottom "
    else:
        description += "middle"

    if result['Circularity'] > 0.7:
        description += " more circular"
    else:
        description += " less circular"
    description += " with " + result['shape'].lower() + " shape"
    description += " and " + result["echo_pattern"].lower() + " echo pattern."

    # Describe intensity
    if result['mean_intensity'] > 80:
        description += " Overall brightness of the image is high."
    else:
        description += " Overall brightness of the image is low."

    return description


# Processing Single Image Function
image_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/images'
label_path = 'C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test/labels'
image_names = os.listdir(image_path)
data_names = os.listdir(label_path)

# Add logic to read the CSV file with filename and type
cancer_type_df = pd.read_csv('C:/Users/Qiran/Downloads/MedSAM-main/data/BUS_dataset/BUS_CLIP/test.csv')  # Update with actual CSV path

image_names.sort()
data_names.sort()
count = 0

# Store results
results = []
feature_results = []  # For storing features for 'image_result.csv'

for image_name, data_name in zip(image_names, data_names):
    image = os.path.join(image_path, image_name)
    mask = os.path.join(label_path, data_name)

    # Read image and mask
    image = cv2.imread(image)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    # Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Threshold the mask to binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure that contours are found
    if len(contours) == 0:
        print(f"No contours found for {image_name}, skipping.")
        continue

    # Assume the largest contour is the lesion
    lesion_contour = max(contours, key=cv2.contourArea)

    # Get location of the centroid
    M = cv2.moments(lesion_contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
    else:
        centroid_x, centroid_y = gray_image.shape[0] // 2, gray_image.shape[1] // 2

    # Calculate all features
    shape, aspect_ratio, circularity = compute_shape_features(lesion_contour)
    posterior_feature, mean_intensity, std_intensity, roi = compute_posterior_acoustic_features(lesion_contour,
                                                                                                gray_image)
    smoothness, spikiness = compute_margin_features(lesion_contour)
    echo_pattern, contrast, correlation, energy, homogeneity = compute_glcm_features(lesion_contour, gray_image)
    lbd = compute_lesion_boundary_feature(lesion_contour, gray_image, k=5)

    # Find the type (0: benign, 1: malignant) for the current image
    cancer_type_row = cancer_type_df[cancer_type_df['filename'] == image_name]
    cancer_type = cancer_type_row['type'].values[0]  # Extract the type (benign/malignant)


    # Store the result in a dictionary for image_result.csv
    feature_result = {
        'filename': str(image_name),
        'shape': shape,
        'Aspect Ratio': aspect_ratio,
        'Circularity': circularity,
        'posterior_feature': posterior_feature,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'smoothness': smoothness,
        'spikiness': spikiness,
        'echo_pattern': echo_pattern,
        'Contrast': contrast,
        'Correlation': correlation,
        'Energy': energy,
        'Homogeneity': homogeneity,
        'lbd': lbd,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "size": gray_image.shape
    }

    feature_results.append(feature_result)

    # Append to the list for image_descriptions.csv
    results.append({"filename": image_name, "description": generate_description(feature_result,cancer_type)})

    # count += 1
    # if count == 100:
    #     break

# Save the descriptions to image_descriptions.csv
description_df = pd.DataFrame(results)
description_df.to_csv('image_descriptions.csv', index=False)

# Save the features to image_result.csv
feature_df = pd.DataFrame(feature_results)
feature_df.to_csv('image_result.csv', index=False)

print("Processing complete. Results saved to 'image_descriptions.csv' and 'image_result.csv'.")



