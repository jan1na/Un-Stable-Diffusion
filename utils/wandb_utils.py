import wandb
from typing import List

run_obj = None


def start(name: str):
    """
    Init wandb.

    :param name: name of wandb upload
    """
    global run_obj
    run_obj = wandb.init(project="stable-diffusion", name=name)


def end():
    """
    Finish wandb.

    """
    wandb.finish()
    global run_obj
    run_obj = None


def upload_images(title: str, images: List, prompts: List[str]):
    """
    Upload images with prompt to wandb.

    :param images: list of images
    :param prompts: list of prompts
    :param title: title of images
    """
    logs = []
    for img, prmt in zip(images, prompts):
        logs.append(wandb.Image(img, caption=prmt))
    wandb.log({title: logs})


def unite_lists(list_of_lists: List[List], num_of_elements: int) -> List:
    """
    Create a new list with alternating values form the lists in list_of_lists.

    :param list_of_lists: list of lists that needs to be combined
    :param num_of_elements: number per list that gets added
    :return: united list of alternating values
    """
    assembled_list = []
    if num_of_elements > len(list_of_lists[0]):
        num_of_elements = len(list_of_lists[0])
    for i in range(num_of_elements):
        for j in range(len(list_of_lists)):
            assembled_list.append(list_of_lists[j][i])
    return assembled_list


def sort_list_by_index(list: List, indexes: List[int]):
    """
    Sort list by list of indexes.

    :param list: list that gets sorted
    :param indexes: indexes that sort the list
    :return: sorted list
    """
    return [list[i] for i in indexes]


def upload_value(title: str, value: float):
    """
    Upload value to wandb.

    :param value: value to upload
    :param title: title describing the value
    """
    wandb.log({title: value})
    run_obj.summary[title] = value


def upload_histogram(title: str, columns_name: str, values: List):
    """
    Create histogram from the values.

    :param title: title of the histogram
    :param columns_name: name of columns
    :param values: list of values for the histogram
    """
    data = [[i] for i in values]
    table = wandb.Table(data=data, columns=[columns_name])
    wandb.log({columns_name + '_histogram': wandb.plot.histogram(table, columns_name, title=title)})
    # TODO: test summary for wandb table
    """
    run_obj.summary["histogram"] = wandb.plot.histogram(table, columns_name, title=title)
    run_obj.summary["hist_table"] = table
    """



