from PIL import Image
import glob
from typing import List
from progress_bar_utils import printProgressBar


def save_list_to_file(values: List, file_path: str):
    """
    Save list of strings into a file.

    :param values: list of strings
    :param file_path: path to the file
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(values))


def load_list_from_file(path: str) -> List[str]:
    """
    Load list of strings from a file.

    :param path: path to the file
    :return: list of strings
    """
    # TODO: some chapters have a dot at the end, some not. Some have whitespace at the end too. Maybe delete
    with open(path) as f:
        values = f.read().splitlines()
    return values


def load_images_from_path(path: str) -> List:
    """
    Load images from a folder into a list.

    :param path: path to the folder where the images are saved
    :return: list of the images
    """ 
    image_list = []
    images = sorted(glob.glob(path + '*.png'))
    printProgressBar(0, len(images), prefix='Load images:')
    for i, filename in enumerate(images):
        image_list.append(Image.open(filename))
        printProgressBar(i + 1, len(images), prefix='Load images:')

    return image_list

    # return [Image.open(filename) for filename in sorted(glob.glob(path + '*.png'))]

