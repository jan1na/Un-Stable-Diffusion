import wandb
from PIL import Image
import glob
from utils.file_utils import read_list_from_file
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score

IMAGES_SAVED = 4


def upload_image_to_wandb(original_images, permuation_images, original_prompts, permutation_prompts, title):
    logs = []
    print(type(original_images))
    print(type(original_prompts))
    for img_o, img_p, prmt_o, prmt_p in zip(original_images, permuation_images, original_prompts, permutation_prompts):
        logs.append(wandb.Image(img_o, caption=prmt_o))
        logs.append(wandb.Image(img_p, caption=prmt_p))
    wandb.log({title: logs})


def load_images_from_path(path: str):
    folder = path + '*.png'
    image_list = []
    for filename in sorted(glob.glob(folder)):
        im = Image.open(filename)
        image_list.append(im)
    return image_list


def create_wandb_doc(init_name: str, image_title: str, original_prompts, permutation_prompts, original_images, permutation_images):
    wandb.init(project="stable-diffusion", name=init_name)

    original_prompts = read_list_from_file('./original_prompts.txt')
    permutation_prompts = read_list_from_file('./permutation_prompts.txt')

    original_images = load_images_from_path('./original_image_outputs/')
    permutation_images = load_images_from_path('./permutation_image_outputs/')

    # save images
    upload_image_to_wandb(original_images[:IMAGES_SAVED], permutation_images[:IMAGES_SAVED], original_prompts[:IMAGES_SAVED],
               permutation_prompts[:IMAGES_SAVED], image_title)

    # cosine similarity
    mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)

    wandb.summary['Mean Cosine Similarity'] = mean_cos_sim

    data = [[i] for i in cos_sim_list]
    table = wandb.Table(data=data, columns=["cosine similarty"])
    wandb.log({'cosine_similarity_histogram': wandb.plot.histogram(table, "cosine similarty", title="Image Cosine Similarity")})

    # clean fid
    wandb.log({"Clean FID Score": clean_fid_score('/home/ml-jfritzke/Stable-Diffusion-BA/original_image_outputs', 
    '/home/ml-jfritzke/Stable-Diffusion-BA/permutation_image_outputs')})


    wandb.finish()



def main():
    original_prompts = read_list_from_file('./original_prompts.txt')
    permutation_prompts = read_list_from_file('./permutation_prompts.txt')
    original_images = load_images_from_path('./original_image_outputs/')
    permutation_images = load_images_from_path('./permutation_image_outputs/')

    create_wandb_doc("naive-char-permutation", 'Naive Char Permutation' original_prompts, permutation_prompts, original_images, permutation_images)



if __name__ == '__main__':
    main()
