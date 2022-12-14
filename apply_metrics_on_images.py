from PIL import Image
import glob
from utils.file_utils import load_list_from_file, load_images_from_path
from metrics.image_metrics import image_array_cosine_similarity, clean_fid_score
from utils.wandb_utils import *

IMAGES_SAVED = 5
IMAGE_PATH = './image_outputs'
PROMPT_PATH = './permutations'


def create_wandb_doc(attack_names: str, attack_file_names: List[str], image_title: str,
                     original_prompts: List, original_images: List, original_control_images: List):

    start(attack_names)

    permutation_prompts = [load_list_from_file(PROMPT_PATH + '/' + file_name + '_prompts.txt') for file_name in
                           attack_file_names]
    permutation_images = [load_images_from_path(IMAGE_PATH + '/' + file_name + '_images/') for file_name in
                          attack_file_names]

    # save images
    save_images = [original_images[:IMAGES_SAVED], original_control_images[:IMAGES_SAVED]]
    for images in permutation_images:
        save_images.append(images[:IMAGES_SAVED])

    save_prompts = [original_prompts[:IMAGES_SAVED], original_prompts[:IMAGES_SAVED]]
    for prompts in permutation_prompts:
        save_prompts.append(prompts[:IMAGES_SAVED])

    """
    upload_images(image_title,
                  unite_lists([original_images[:IMAGES_SAVED],
                               original_control_images[:IMAGES_SAVED],
                               permutation_images[:IMAGES_SAVED]]),
                  unite_lists([original_prompts[:IMAGES_SAVED],
                               original_prompts[:IMAGES_SAVED],
                               permutation_prompts[:IMAGES_SAVED]]))
    """

    upload_images(image_title, unite_lists(save_images), unite_lists(save_prompts))

    # control
    mean_cos_sim_control, cos_sim_list_control = image_array_cosine_similarity(original_images, original_control_images)
    upload_value('Control: Mean Cosine Similarity', mean_cos_sim_control)
    upload_histogram("Control: Image Cosine Similarity", "cosine similarity control", cos_sim_list_control)

    # Cosine Similarity
    for perm_images, name in zip(permutation_prompts, attack_file_names):
        mean_cos_sim, cos_sim_list = image_array_cosine_similarity(original_images, perm_images)
        upload_value('Mean Cosine Similarity: ' + name, mean_cos_sim)
        upload_histogram("Image Cosine Similarity: " + name, "cosine similarity", cos_sim_list)

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

    attack_file_names = ["naive_char", "char", "delete_char", "duplicate_char"]
    # image_titles = ['Naive Char Permutation', 'Char Permutation', 'Delete Char Permutation',
    # 'Duplicate Char Permutation']

    create_wandb_doc('adversarial text attacks', attack_file_names, 'Adversarial Text Attacks', original_prompts,
                     original_images, original_control_images)


if __name__ == '__main__':
    main()
