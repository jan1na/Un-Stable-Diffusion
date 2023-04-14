import os
import clip
import torch
from typing import List, Tuple
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
import glob
import lpips
from cleanfid import fid
from PIL import Image
from numpy import ndarray
from utils.file_utils import load_list_from_file

# Load the model for cosine similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device)

import gc
from numba import cuda



def image_cosine_similarity(image_0, image_1) -> float:
    """
    Cosine similarity between two images.

    :param image_0: first image
    :param image_1: second image
    :return: cosine similarity
    """

    image_input_0 = preprocess(image_0).unsqueeze(0).to(device)
    image_input_1 = preprocess(image_1).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features_0 = clip_model.encode_image(image_input_0)
        image_features_1 = clip_model.encode_image(image_input_1)

    image_features_0 /= image_features_0.norm(dim=-1, keepdim=True)
    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    cosine_similarity = image_features_0 @ image_features_1.T

    return cosine_similarity


def image_array_cosine_similarity(image_list_0: List, image_list_1: List) -> [float, List[float]]:
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


def clean_fid_score(image_folder_0: str, image_folder_1: str) -> float:
    """
    Calculate the clean FID score for 2 image directories.

    :param image_folder_0: image directory 0
    :param image_folder_1: image directory 1
    :return: fid score
    """
    score = fid.compute_fid(image_folder_0, image_folder_1, mode="clean")
    return score


def perceptual_similarity(image_0, image_1):
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    # TODO: convert PNG to RGB
    rgb_image_0 = image_0.convert('RGB')
    rgb_image_1 = image_1.convert('RGB')

    # normalize to [-1,1]
    rgb_image_0 = np.array(rgb_image_0)
    rgb_image_1 = np.array(rgb_image_1)

    normalized_image_0 = (rgb_image_0.astype(np.float32) / 255.0) * 2.0 - 1.0
    normalized_image_1 = (rgb_image_1.astype(np.float32) / 255.0) * 2.0 - 1.0

    img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1, 3, 64, 64)
    d = loss_fn_alex(normalized_image_0, normalized_image_1)
    return d


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()


def image_content_similarity(captions_path_0: str, captions_path_1: str) -> [float, List[float]]:
    captions_0 = load_list_from_file(captions_path_0)
    captions_1 = load_list_from_file(captions_path_1)

    cos_sim = []
    print(len(captions_0))

    batch_size = 50
    iterations = len(captions_0) // batch_size if len(captions_0) % batch_size == 0 else len(captions_0) // batch_size + 1
    for i in range(iterations):
        cos_sim += ics_batch(captions_0[i * batch_size: (i+1) * batch_size], captions_1[i * batch_size: (i+1) * batch_size])
    return np.mean(cos_sim), cos_sim


def ics_batch(captions_0, captions_1):
    cos_sim = []
    for caption_0, caption_1 in zip(captions_0, captions_1):

        text_input = tokenizer([caption_0, caption_1],
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]

        caption_0_feature = torch.flatten(text_embeddings[0].unsqueeze(0), start_dim=1)
        caption_1_feature = torch.flatten(text_embeddings[1].unsqueeze(0), start_dim=1)
        cos_sim.append(cosine_similarity(caption_0_feature, caption_1_feature))

    cos_sim = [x.item() for x in cos_sim]
    return cos_sim


def image_content_similarity_old(captions_path_0: str, captions_path_1: str) -> [float, List[float]]:
    captions_0 = load_list_from_file(captions_path_0)
    captions_1 = load_list_from_file(captions_path_1)

    cos_sim = []

    i=0
    for caption_0, caption_1 in zip(captions_0, captions_1):
        print("iteration:", i)
        i+=1

        text_input = tokenizer([caption_0, caption_1],
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]

        caption_0_feature = torch.flatten(text_embeddings[0].unsqueeze(0), start_dim=1)
        caption_1_feature = torch.flatten(text_embeddings[1].unsqueeze(0), start_dim=1)
        cos_sim.append(cosine_similarity(caption_0_feature, caption_1_feature))

        # Clear GPU memory
        del text_embeddings, caption_0_feature, caption_1_feature, text_input
        torch.cuda.empty_cache()
        gc.collect()
    cos_sim = [x.item() for x in cos_sim]
    return np.mean(cos_sim), cos_sim


def image_prompt_similarity(images: List, prompts: List[str]) -> [float, List[float]]:
    cos_sim = []

    for img, prompt in zip(images[:10], prompts[:10]):
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            # image_features = clip_model.encode_image(image)
            # text_features = clip_model.encode_text(text)

            logits_per_image, logits_per_text = clip_model(image, text)
            print("image:", logits_per_image)
            print("text:", logits_per_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            cos_sim.append(logits_per_image)

            print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    return np.mean(cos_sim), cos_sim
