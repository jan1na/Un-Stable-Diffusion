import numpy as np

from utils.file_utils import load_list_from_file, load_images_from_path
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score, image_content_similarity
from utils.wandb_utils import *
from attack_types import file_names, run_names, title_names, IMAGES_SAVED, IMAGE_PATH, PROMPT_PATH, CAPTION_PATH
from rtpt import RTPT


rtpt = RTPT('JF', 'metric_application', 10)


def create_wandb_doc(run_name: str, attack_file_name: str, image_title: str, original_prompts: List[str],
                     original_images: List, sorted_by_cosine_similarity: bool = False,
                     sorted_by_caption_similarity: bool = False):
    """
    Upload the images and metric results as single values and histograms to wandb.

    :param run_name: name of the wandb run
    :param attack_file_name: name of file with attack prompts
    :param image_title: title of uploaded images
    :param original_prompts: original prompts
    :param original_images: original images
    :param sorted_by_cosine_similarity: sort the images by cosine similarity from worst to best
    :param sorted_by_caption_similarity: sort the images by caption similarity from worst to best
    """

    start(run_name)
    print("wandb started")
    print("run:", run_name)

    ORIGINAL_IMAGE_PATH = IMAGE_PATH + '/original_images/'
    ATTACK_IMAGE_PATH = IMAGE_PATH + '/' + attack_file_name + '_images/'

    """
    permutation_prompts = load_list_from_file(PROMPT_PATH + '/' + attack_file_name + '_prompts.txt')
    permutation_images = load_images_from_path(IMAGE_PATH + '/' + attack_file_name + '_images/')

    # Cosine Similarity
    print("calc Cosine Similarity")

    mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, permutation_images)
    upload_value('Mean Cosine Similarity', mean_cos_sim)
    upload_histogram("Image Cosine Similarity", "cosine similarity", cos_sim_list)

    # Clean FID
    print("calc Clean FID")
    upload_value("Clean FID Score", clean_fid_score(ORIGINAL_IMAGE_PATH, ATTACK_IMAGE_PATH))

    """
    # Image Caption Similarity
    print("calc Image Caption Similarity")
    mean_img_cap_sim, img_cap_sim_list = image_content_similarity(CAPTION_PATH + '/original',
                                                                  CAPTION_PATH + '/' + attack_file_name)
    upload_value('Image Caption Similarity', mean_img_cap_sim)
    upload_histogram("Image Caption Similarity", "image caption cosine similarity", img_cap_sim_list)

    # upload images to wandb sometimes sorted by a metric
    """

    if sorted_by_cosine_similarity:
        indexes = list(np.argsort(cos_sim_list))
    elif sorted_by_caption_similarity:
        indexes = list(np.argsort(img_cap_sim_list))
    else:
        indexes = list(np.arange(len(original_prompts)))

    image_list = [sort_list_by_index(original_images, indexes), sort_list_by_index(permutation_images, indexes)]
    prompt_list = [sort_list_by_index(original_prompts, indexes), sort_list_by_index(permutation_prompts, indexes)]

    upload_images(image_title, unite_lists(image_list, IMAGES_SAVED), unite_lists(prompt_list, IMAGES_SAVED))
    """

    end()


def main():
    rtpt.start()

    original_prompts = load_list_from_file(PROMPT_PATH + '/original_prompts.txt')
    original_images = load_images_from_path(IMAGE_PATH + '/original_images/')

    for file_name, run_name, image_title in zip(file_names, run_names, title_names):
        create_wandb_doc(run_name, file_name, image_title, original_prompts, original_images, False, True)
        rtpt.step()


if __name__ == '__main__':
    main()
