import wandb
from PIL import Image
import glob
from utils.file_utils import read_list_from_file



wandb.init(project="stable-diffusion")

def save_image(original_images, permuation_images, original_prompts, permutation_prompts, title):
    logs = []
    print(type(original_images))
    print(type(original_prompts))
    for img_o, img_p, prmt_o, prmt_p in zip(original_images, permuation_images, original_prompts, permutation_prompts):
        logs.append(wandb.Image(img_o, caption=prmt_o))
        logs.append(wandb.Image(img_p, caption=prmt_p))
    wandb.log({title: logs})


def load_images(path: str):
    folder = path + '*.png'
    image_list = []
    for filename in sorted(glob.glob(folder)):
        im=Image.open(filename)
        image_list.append(im)
    return image_list


def main():
    original_prompts = read_list_from_file('./original_prompts.txt')
    permutation_prompts = read_list_from_file('./permutation_prompts.txt')
    original_images = load_images('./original_image_outputs/')
    permutation_images = load_images('./permutation_image_outputs/')
    save_image(original_images, permutation_images, original_prompts, permutation_prompts, 'naive char permuation')


if __name__ == '__main__':
    main()



wandb.finish()