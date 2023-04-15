import clip
import torch
from typing import List
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn.functional import cosine_similarity
from cleanfid import fid
from utils.file_utils import load_list_from_file
from utils.progress_bar_utils import printProgressBar

# Load the model for cosine similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device)


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
    cos_sim = image_features_0 @ image_features_1.T

    return cos_sim


def image_array_cosine_similarity(image_list_0: List, image_list_1: List) -> [float, List[float]]:
    """
    Cosine similarity of two lists of images for each of the images in the two lists.

    :param image_list_0: first list of images
    :param image_list_1: second list of images
    :return: list of cosine similarities between every element in the image lists
    """
    cos_sim_list = []
    printProgressBar(0, len(image_list_0), prefix='Image Cosine Sim:')
    i = 0
    for img_0, img_1 in zip(image_list_0, image_list_1):
        cos_sim_list.append(image_cosine_similarity(img_0, img_1)[0][0].item())
        printProgressBar(i + 1, len(image_list_0), prefix='Image Cosine Sim:')
        i += 1
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


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()


def image_content_similarity(captions_path_0: str, captions_path_1: str) -> [float, List[float]]:
    captions_0 = load_list_from_file(captions_path_0)
    captions_1 = load_list_from_file(captions_path_1)

    cos_sim = []

    batch_size = 50
    iterations = len(captions_0) // batch_size if len(captions_0) % batch_size == 0 else len(
        captions_0) // batch_size + 1
    printProgressBar(0, iterations, prefix='Image Content Sim:')
    for i in range(iterations):
        cos_sim += ics_batch(captions_0[i * batch_size: (i + 1) * batch_size],
                             captions_1[i * batch_size: (i + 1) * batch_size])
        printProgressBar(i + 1, iterations, prefix='Image Content Sim:')
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


def image_prompt_similarity(images: List, prompts: List[str]) -> [float, List[float]]:
    cos_sim = []
    printProgressBar(0, len(images), prefix='Image Prompt Sim:')
    i = 0

    for img, prompt in zip(images, prompts):
        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            cos_sim.append(cosine_similarity(image_features, text_features).item())
        printProgressBar(i + 1, len(images), prefix='Image Prompt Sim:')
        i += 1
    return np.mean(cos_sim), cos_sim
