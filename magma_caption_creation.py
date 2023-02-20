from utils.wandb_utils import *
from attack_types import file_names, run_names
from transformers import CLIPTextModel, CLIPTokenizer
from magma import Magma
from magma.image_input import ImageInput
import glob
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from utils.file_utils import save_list_to_file


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


def image_content_similarity(image_folder):
    captions = []

    for image_path in sorted(glob.glob(image_folder + '*.png')):
        captions.append(get_image_caption(image_path))





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



    ORIGINAL_IMAGE_PATH = IMAGE_PATH + '/original_images/'
    ATTACK_IMAGE_PATH = IMAGE_PATH + '/' + attack_file_name + '_images/'

    image_content_similarity(ATTACK_IMAGE_PATH)


def main():
    print("in main")
    image_content_similarity(IMAGE_PATH + '/original_images/')

    for file_name, run_name in zip(file_names, run_names):
        print("filename", file_name)
        image_content_similarity(IMAGE_PATH + '/' + file_name + '_images/')


if __name__ == '__main__':
    main()
