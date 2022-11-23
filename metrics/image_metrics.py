import os
import clip
import torch
from typing import List

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def image_cosine_similarity(image_array_0: list, image_array_1: list) -> list(float):
    # Calculate features
    with torch.no_grad():
        image_features_0 = model.encode_image(image_array_0)
        image_features_1 = model.encode_image(image_array_1)
    
    image_features_0 /= image_features_0.norm(dim=-1, keepdim=True)
    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    cosine_similarity = 100.0 * image_features @ text_features.T
    print("consine similarity:", cosine_similarity)
    return cosine_similarity