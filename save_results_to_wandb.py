import wandb
from PIL import Image
import glob
from utils.file_utils import read_list_from_file
from metrics.image_metrics import image_cosine_similarity, image_array_cosine_similarity


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
    mean_cos_sim , cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)
    wandb.log({"Mean cosine similarity of the whole dataset.": mean_cos_sim })
    data = [[s] for s in cos_sim_list]
    table = wandb.Table(data=data, columns=["scores"])
    wandb.log({"histogram": wandb.plot.histogram(table, "scores", title="Cosine similarity between the both images created out of the original prompt and the permuted prompt.")})
        


if __name__ == '__main__':
    main()



wandb.finish()