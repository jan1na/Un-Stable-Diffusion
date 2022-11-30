from PIL import Image
import glob


def save_list_to_file(values, file_path):
    """
    Save list of strings into a file.

    :param values: list of strings
    :param file_path: path to the file
    """
    with open(file_path, 'w') as fp:
        fp.write(''.join(values))


def load_list_from_file(path: str):
    """
    Load list of strings from a file.

    :param path: path to the file
    :return: list of strings
    """
    with open(path) as f:
        values = f.readlines()
    return values


def load_images_from_path(path: str):
    """
    Load images from a folder into a list.

    :param path: path to the folder where the images are saved
    :return: list of the images
    """
    return [Image.open(filename) for filename in sorted(glob.glob(path + '*.png'))]

