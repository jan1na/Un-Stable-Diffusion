from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from attack_types import file_names, run_names, PROMPT_NUMBER
from utils.file_utils import save_list_to_file, load_images_from_path
from typing import List
from rtpt import RTPT

IMAGE_PATH = './image_outputs'
CAPTION_PATH = './image_captions'

rtpt = RTPT('JF', 'caption_creation', PROMPT_NUMBER * 11)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def get_image_caption(image) -> str:
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def get_image_captions(image_folder) -> List[str]:
    captions = []

    images = load_images_from_path(image_folder)
    for image in images:
        captions.append(get_image_caption(image))
        rtpt.step()
    print("captions:", len(captions))
    return captions


def main():
    rtpt.start()

    captions = get_image_captions(IMAGE_PATH + '/original_images/')
    save_list_to_file(captions, CAPTION_PATH + '/original')

    for file_name, run_name in zip(file_names, run_names):
        captions = get_image_captions(IMAGE_PATH + '/' + file_name + '_images/')
        save_list_to_file(captions, CAPTION_PATH + '/' + file_name)
        print("created file:", file_name)


if __name__ == '__main__':
    main()
