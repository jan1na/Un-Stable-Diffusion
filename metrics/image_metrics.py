import os
import clip
import torch
from typing import List, Tuple, Any
import numpy as np
from numpy import ndarray

# Load the model for cosine similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


def image_cosine_similarity(image_0, image_1) -> float:
    """
    Cosine similarity between two images.

    :param image_0: first image
    :param image_1: second image
    :return: cosine similarity
    """

    print("image_0", image_0.shape)

    image_input_0 = preprocess(image_0).unsqueeze(0).to(device)
    image_input_1 = preprocess(image_1).unsqueeze(0).to(device)

    print("image_input_0", image_input_0.shape)

    # Calculate features
    with torch.no_grad():
        image_features_0 = model.encode_image(image_input_0)
        image_features_1 = model.encode_image(image_input_1)

    print("image_features_0", image_features_0.shape)
    
    image_features_0 /= image_features_0.norm(dim=-1, keepdim=True)
    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    cosine_similarity = image_features_0 @ image_features_1.T

    print("cosine_similarity", cosine_similarity.shape)
    return cosine_similarity


def image_array_cosine_similarity(image_list_0: List, image_list_1: List) -> tuple[ndarray, list[np.ndarray]]:
    """
    Cosine similarity of two lists of images for each of the images in the two lists.

    :param image_list_0: first list of images
    :param image_list_1: second list of images
    :return: list of cosine similarities between every element in the image lists
    """
    cos_sim_list = []
    for img_0, img_1 in zip(image_list_0, image_list_1):
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


def get_aesthetic_model(clip_model="vit_l_14"):
    import os
    import torch
    import torch.nn as nn
    from os.path import expanduser  # pylint: disable=import-outside-toplevel
    from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m