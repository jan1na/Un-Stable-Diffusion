from utils.wandb_utils import *
from attack_types import file_names, run_names
from transformers import CLIPTextModel, CLIPTokenizer
from magma import Magma
from magma.image_input import ImageInput
import glob
import torch
from torch.nn.functional import cosine_similarity
import numpy as np


IMAGES_SAVED = 10
IMAGE_PATH = './image_outputs'
PROMPT_PATH = './permutations'

magma_model = Magma.from_checkpoint(
    config_path="configs/MAGMA_v1.yml",
    checkpoint_path="./mp_rank_00_model_states.pt",
    device='cuda:0'
)

def get_image_caption(image_path: str) -> str:
    inputs = [
        # supports urls and path/to/image
        ImageInput(image_path),
        'Describe the image:'
    ]

    # returns a tensor of shape: (1, 149, 4096)
    embeddings = magma_model.preprocess_inputs(inputs)

    # returns a list of length embeddings.shape[0] (batch size)
    output = magma_model.generate(
        embeddings=embeddings,
        max_steps=6,
        temperature=0.7,
        top_k=0,
    )
    return output[0]


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()


def image_content_similarity(image_folder_0: str, image_folder_1: str) -> [float, List[float]]:
    cos_sim = []

    for image_path_0, image_path_1 in zip(sorted(glob.glob(image_folder_0 + '*.png')),
                                          sorted(glob.glob(image_folder_1 + '*.png'))):
        caption_0 = get_image_caption(image_path_0)
        caption_1 = get_image_caption(image_path_1)

        text_input = tokenizer([caption_0, caption_1],
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]

        caption_0_feature = torch.flatten(text_embeddings[0].unsqueeze(0), start_dim=1)
        caption_1_feature = torch.flatten(text_embeddings[1], start_dim=1)
        cos_sim.append(cosine_similarity(caption_0_feature, caption_1_feature))
    return np.mean(cos_sim), cos_sim


def create_wandb_doc(run_name: str, attack_file_name: str):
    """
    Upload the images and metric results as single values and histograms to wandb.

    :param run_name: name of the wandb run
    :param attack_file_name: name of file with attack prompts
    :param image_title: title of uploaded images
    :param original_prompts: original prompts
    :param original_images: original images
    :param sorted_by_cosine_similarity: sort the images from worst to best
    """

    start(run_name)
    print("wandb started")

    ORIGINAL_IMAGE_PATH = IMAGE_PATH + '/original_images/'
    ATTACK_IMAGE_PATH = IMAGE_PATH + '/' + attack_file_name + '_images/'

    # Image Caption Similarity
    print("calc image caption similarity")
    mean_cos_sim, cos_sim_list = image_content_similarity(ORIGINAL_IMAGE_PATH, ATTACK_IMAGE_PATH)
    upload_value('Mean Cosine Similarity', mean_cos_sim)
    upload_histogram("Image Cosine Similarity", "cosine similarity", cos_sim_list)

    end()


def main():
    print("in main")

    for file_name, run_name in zip(file_names, run_names):
        print("filename", file_name)
        create_wandb_doc(run_name, file_name)


if __name__ == '__main__':
    main()
