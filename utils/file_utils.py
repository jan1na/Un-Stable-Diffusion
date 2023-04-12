from PIL import Image
import glob
from typing import List


def save_list_to_file(values: List[str], file_path: str):
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
    return [Image.open(filename) for filename in sorted(glob.glob(path + '*.png'))]


def delete_empty_lines(path: str):
    """
    Delete all empty lines in text file.

    :param path: path to text file
    """
    lst = []
    print("_____________________________________________________________________________")
    for line in load_list_from_file(path)[9999:10000]:
        lst.append(line.strip())
        print("'" + line + "'")

    # save_list_to_file(lst, path)
