import os
import torch
import clip
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Load the ResNet-based CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)  # Use ResNet-50 model

# Define the directory containing images
image_dir = "./data/"  # Directory with images
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('jpg', 'png'))]

# User-provided description
description = "a brwon cat lying on a porch"

# Tokenize the description
text_input = clip.tokenize([description]).to(device)

# Preprocess and encode all images in the directory
image_features_list = []
image_names = []
with torch.no_grad():
    for image_path in image_paths:
        # Preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Encode the image and normalize features
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(image_features)
        image_names.append(image_path)

    # Encode the text description and normalize features
    text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarities and find the best match
image_features_tensor = torch.cat(image_features_list, dim=0)
similarities = (image_features_tensor @ text_features.T).squeeze(1).cpu().numpy()

# Find the most similar image
best_match_idx = np.argmax(similarities)
best_match_path = image_names[best_match_idx]

# Print the result
print(f"The image most similar to the description '{description}' is: {best_match_path}")

# Optionally, display the best match
best_image = Image.open(best_match_path)
plt.imshow(best_image)
plt.axis('off')
plt.show()
