import wandb


def start(name: str):
    """
    Init wandb.

    :param name: name of wandb upload
    """
    wandb.init(project="stable-diffusion", name=name)


def end():
    """
    Finish wandb.

    """
    wandb.finish()


def upload_images(title: str, images: list, prompts: list[str]):
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


def unite_lists(list_of_lists: list[list]) -> list:
    """
    Create a new list with alternating values form the lists in list_of_lists.

    :param list_of_lists: list of lists that needs to be combined
    :return: united list of alternating values
    """
    assembled_list = []
    for i in range(len(list_of_lists[0])):
        for j in range(len(list_of_lists)):
            assembled_list.append(list_of_lists[j][i])
    return assembled_list


def upload_value(title: str, value: float):
    """
    Upload value to wandb.

    :param value: value to upload
    :param title: title describing the value
    """
    wandb.summary[title] = value


def upload_histogram(title: str, columns_name: str, values: list):
    """
    Create histogram from the values.

    :param title: title of the histogram
    :param columns_name: name of columns
    :param values: list of values for the histogram
    """
    data = [[i] for i in values]
    table = wandb.Table(data=data, columns=[columns_name])
    wandb.log({columns_name + '_histogram': wandb.plot.histogram(table, columns_name, title=title)})




