from PIL import Image
import glob
from utils.file_utils import load_list_from_file, load_images_from_path
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score
from utils.wandb_utils import *

IMAGES_SAVED = 5


"""
def upload_image_to_wandb(original_images, permuation_images, original_prompts, permutation_prompts, title):
    logs = []
    for img_o, img_p, prmt_o, prmt_p in zip(original_images, permuation_images, original_prompts, permutation_prompts):
        logs.append(wandb.Image(img_o, caption=prmt_o))
        logs.append(wandb.Image(img_p, caption=prmt_p))
    wandb.log({title: logs})
"""


"""
def load_images_from_path(path: str):
    folder = path + '*.png'
    image_list = []
    for filename in sorted(glob.glob(folder)):
        im = Image.open(filename)
        image_list.append(im)
    return image_list
    
"""


def create_wandb_doc(attack_name: str, attack_file_name, image_title: str, original_prompts: list, original_images: list,
                     original_control_images: list):
    start(attack_name)

    permutation_prompts = load_list_from_file('./' + attack_file_name + '_permutation_prompts.txt')
    permutation_images = load_images_from_path('./' + attack_file_name + '_permutation_image_outputs/')

    list1 = unite_lists([original_images[:IMAGES_SAVED],
                               original_control_images[:IMAGES_SAVED],
                               permutation_images[:IMAGES_SAVED]])
    list2 = unite_lists([original_prompts[:IMAGES_SAVED],
                               original_prompts[:IMAGES_SAVED],
                               permutation_prompts[:IMAGES_SAVED]])
    print(list2)
    # save images
    upload_images(image_title,
                  unite_lists([original_images[:IMAGES_SAVED],
                               original_control_images[:IMAGES_SAVED],
                               permutation_images[:IMAGES_SAVED]]),
                  unite_lists([original_prompts[:IMAGES_SAVED],
                               original_prompts[:IMAGES_SAVED],
                               permutation_prompts[:IMAGES_SAVED]]))

    # upload_image_to_wandb(original_images[:IMAGES_SAVED], permutation_images[:IMAGES_SAVED],
    #                      original_prompts[:IMAGES_SAVED],
    #                     permutation_prompts[:IMAGES_SAVED], image_title)

    # Cosine Similarity
    mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)
    # control
    mean_cos_sim_control, cos_sim_list_control = image_array_cosine_similarity(original_images, original_control_images)

    # wandb.summary['Mean Cosine Similarity'] = mean_cos_sim
    upload_value('Mean Cosine Similarity', mean_cos_sim)
    upload_value('Control: Mean Cosine Similarity', mean_cos_sim_control)

    """
    data = [[i] for i in cos_sim_list]
    table = wandb.Table(data=data, columns=["cosine similarity"])
    wandb.log({'cosine_similarity_histogram': wandb.plot.histogram(table, "cosine similarity",
                                                                   title="Image Cosine Similarity")})
    """
    upload_histogram("Image Cosine Similarity", "cosine similarity", cos_sim_list)
    upload_histogram("Control: Image Cosine Similarity", "cosine similarity", cos_sim_list_control)

    # clean fid
    upload_value("Clean FID Score", clean_fid_score('./original_image_outputs',
                                                    './' + attack_file_name + '_permutation_image_outputs/'))
    upload_value("Control: Clean FID Score", clean_fid_score('./original_image_outputs',
                                                             './original_control_image_outputs'))

    end()


def main():
    original_prompts = load_list_from_file('./original_prompts.txt')
    original_images = load_images_from_path('./original_image_outputs/')
    original_control_images = load_images_from_path('./original_control_image_outputs')

    create_wandb_doc("naive-char-permutation", "naive_char_permutation", 'Naive Char Permutation', original_prompts,
                     original_images, original_control_images)


if __name__ == '__main__':
    main()
