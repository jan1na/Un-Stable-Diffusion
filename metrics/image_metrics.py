import os
import clip
import torch
from typing import List
import numpy as np

# Load the model for cosine similarity
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
        cos_sim_list.append(image_cosine_similarity(img_0, img_1)[0][0].item())
    return np.mean(cos_sim_list), cos_sim_list


def clean_fid_score(image_folder_0, image_folder_1):
    from cleanfid import fid
    score = fid.compute_fid(image_folder_0, image_folder_1)


def perceptual_similarity():
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    # TODO: convert PNG to RGB 

    import torch
    img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1,3,64,64)
    d = loss_fn_alex(img0, img1)