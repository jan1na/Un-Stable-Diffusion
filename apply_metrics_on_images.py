import numpy as np

from utils.file_utils import load_list_from_file, load_images_from_path
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score
from utils.wandb_utils import *

IMAGES_SAVED = 20
IMAGE_PATH = './image_outputs'
PROMPT_PATH = './permutations'


def create_wandb_doc(run_name: str, attack_file_name: str, image_title: str, original_prompts: List[str],
                     original_images: List, sorted_by_cosine_similarity: bool = False):
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

    permutation_prompts = load_list_from_file(PROMPT_PATH + '/' + attack_file_name + '_prompts.txt')
    permutation_images = load_images_from_path(IMAGE_PATH + '/' + attack_file_name + '_images/')

    # Cosine Similarity
    mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)
    upload_value('Mean Cosine Similarity', mean_cos_sim)
    upload_histogram("Image Cosine Similarity", "cosine similarity", cos_sim_list)

    # clean fid
    # upload_value("Clean FID Score", clean_fid_score('./original_image_outputs',
    #                                                 './' + attack_file_names + '_permutation_image_outputs/'))
    # upload_value("Control: Clean FID Score", clean_fid_score('./original_image_outputs',
    #                                                          './original_control_image_outputs'))

    # upload images to wandb sometimes sorted by a metric

    if sorted_by_cosine_similarity:
        indexes = np.argsort(cos_sim_list)
    else:
        indexes = np.arange(len(original_prompts))

    image_list = [original_images[indexes], permutation_images[indexes]]
    prompt_list = [original_prompts[indexes], permutation_prompts[indexes]]

    print(prompt_list)

    upload_images(image_title,
                  unite_lists(image_list, IMAGES_SAVED),
                  unite_lists(prompt_list, IMAGES_SAVED))

    end()


def main():
    original_prompts = load_list_from_file(PROMPT_PATH + '/original_prompts.txt')
    original_images = load_images_from_path(IMAGE_PATH + '/original_images/')

    attack_file_names = ["original_control", "naive_char", "char", "delete_char", "duplicate_char", "synonym_word",
                         "homophone_word"]
    run_names = ["original-control", "naive-char", "char", "delete-char", "duplicate-char", "synonym-word",
                 "homophone-word"]
    image_titles = ['Original Control', 'Naive Char Permutation', 'Char Permutation', 'Delete Char Permutation',
                    'Duplicate Char Permutation', 'Synonym Word Permutation', 'Homophone Word Permutation']

    for file_name, run_name, image_title in zip(attack_file_names, run_names, image_titles):
        create_wandb_doc(run_name, file_name, image_title, original_prompts, original_images, True)


if __name__ == '__main__':
    main()
