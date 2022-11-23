import os
import clip
import torch
from typing import List

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def image_cosine_similarity(image_0, image_1) -> float:

    image_input_0 = preprocess(image_0).unsqueeze(0).to(device)
    image_input_1 = preprocess(image_1).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features_0 = model.encode_image(image_input_0)
        image_features_1 = model.encode_image(image_input_1)
    
    image_features_0 /= image_features_0.norm(dim=-1, keepdim=True)
    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    cosine_similarity = image_features_0 @ image_features_1.T
    print("consine similarity:", cosine_similarity)
    return cosine_similarity


def image_array_cosine_similarity(image_array_0, image_array_1):
    cos_sim_list = []
    for img_0, img_1 in zip(image_array_0, image_array_1):
        cos_sim_list.append(image_cosine_similarity(img_0, img_1))
    return torch.mean(torch.stack(cos_sim_list)), cos_sim_list
