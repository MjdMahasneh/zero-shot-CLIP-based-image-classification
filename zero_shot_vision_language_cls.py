import torch
import clip
from PIL import Image

# Load the pre-trained CLIP model and the tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open("./data/image_1.jpg")).unsqueeze(0).to(device)

# Define the class names
class_names = ["a cat", "a dog", "a car", "a tree"]

# Tokenize the class names
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in class_names]).to(device)

# Forward pass through the model
with torch.no_grad():
    image_features = model.encode_image(image)
    print('shape of image_features:', image_features.shape)
    text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores
    similarities = (image_features @ text_features.T).squeeze(0) ## compute the dot product, and `.T` transposes the text feature matrix for compatibility.

# Get the most probable class
probs = similarities.softmax(dim=-1).cpu().numpy()
predicted_class = class_names[probs.argmax()]

print(f"Predicted class: {predicted_class}")


