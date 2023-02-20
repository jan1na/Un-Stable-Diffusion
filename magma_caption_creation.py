from attack_types import file_names, run_names
from magma import Magma
from magma.image_input import ImageInput
import glob
from utils.file_utils import save_list_to_file
from typing import List

IMAGES_SAVED = 10
IMAGE_PATH = './image_outputs'
PROMPT_PATH = './permutations'
CAPTION_PATH = './image_captions'

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
        max_steps=12,
        temperature=0.7,
        top_k=0,
    )
    return output[0]


def get_image_captions(image_folder) -> List[str]:
    captions = []

    for image_path in sorted(glob.glob(image_folder + '*.png')):
        captions.append(get_image_caption(image_path))
    return captions


def main():
    print("in main")
    captions = get_image_captions(IMAGE_PATH + '/original_images/')
    save_list_to_file(captions, CAPTION_PATH + '/original')

    for file_name, run_name in zip(file_names, run_names):
        print("filename", file_name)
        captions = get_image_captions(IMAGE_PATH + '/' + file_name + '_images/')
        save_list_to_file(captions, CAPTION_PATH + '/' + file_name)


if __name__ == '__main__':
    main()
