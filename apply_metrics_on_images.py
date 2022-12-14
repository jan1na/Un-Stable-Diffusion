from PIL import Image
import glob
from utils.file_utils import load_list_from_file, load_images_from_path
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score
from utils.wandb_utils import *

IMAGES_SAVED = 5
IMAGE_PATH = './image_outputs'
PROMPT_PATH = './permutations'


def create_wandb_doc(attack_name: str, attack_file_name: str, image_title: str,
                     original_prompts: List, original_images: List):

    start(attack_name)

    permutation_prompts = load_list_from_file(PROMPT_PATH + '/' + attack_file_name + '_prompts.txt')
    permutation_images = load_images_from_path(IMAGE_PATH + '/' + attack_file_name + '_images/')

    upload_images(image_title,
                  unite_lists([original_images[:IMAGES_SAVED],
                               permutation_images[:IMAGES_SAVED]]),
                  unite_lists([original_prompts[:IMAGES_SAVED],
                               permutation_prompts[:IMAGES_SAVED]]))

    # Cosine Similarity
    mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)
    upload_value('Mean Cosine Similarity: ' + attack_file_name, mean_cos_sim)
    upload_histogram("Image Cosine Similarity: " + attack_file_name, "cosine similarity", cos_sim_list)

    # clean fid
    # upload_value("Clean FID Score", clean_fid_score('./original_image_outputs',
    #                                                 './' + attack_file_names + '_permutation_image_outputs/'))
    # upload_value("Control: Clean FID Score", clean_fid_score('./original_image_outputs',
    #                                                          './original_control_image_outputs'))

    end()


def main():

    original_prompts = load_list_from_file(PROMPT_PATH + '/original_prompts.txt')
    original_images = load_images_from_path(IMAGE_PATH + '/original_images/')
    original_control_images = load_images_from_path(IMAGE_PATH + '/original_control_images/')

    attack_file_names = ["original_control", "naive_char", "char", "delete_char", "duplicate_char"]
    attack_names = ["original-control", "naive-char", "char", "delete-char", "duplicate-char"]
    image_titles = ['Original Control', 'Naive Char Permutation', 'Char Permutation', 'Delete Char Permutation',
                    'Duplicate Char Permutation']

    for file_name, attack_name, image_title in zip(attack_file_names, attack_names, image_titles):
        create_wandb_doc(attack_name, file_name, image_title, original_prompts, original_images)


if __name__ == '__main__':
    main()
