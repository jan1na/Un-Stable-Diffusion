import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from typing import List

WANDB_DIR = "./wandb/csv-files/"


def create_parser():
    """
    Define python script argument parser.

    :return: arguments
    """
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument('-e',
                        '--metric',
                        default="cosine-similarity",
                        type=str,
                        dest="metric",
                        help='metric of the processed data')
    args = parser.parse_args()
    return args


def read_csv_data() -> [np.ndarray, List[str]]:
    """
    Read csv data into a numpy array and a list.

    :return: data numpy array and label list
    """
    directory = os.fsencode(WANDB_DIR)
    data, label = [], []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename not in ['synonym-word', 'typo-char', 'naive-char', 'original-control']:
            continue
        label.append(filename)
        with open(WANDB_DIR + filename) as f:
            data.append([float(v[1:-1]) for v in f.read().splitlines()[1:]])
    return np.asarray(data), label


def plot_histogram(data: np.ndarray, label: List[str], title: str):
    """
    Plot data in a histogram and save the plot.

    :param data: data that gets plotted
    :param label: label for the data
    :param title: title for the whole plot
    """
    plt.hist(data.T, 7, density=True, histtype='bar', label=label)
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig('./wandb/plots/' + title + '.png')
    plt.show()
    plt.close()


def main():
    args = create_parser()
    data, label = read_csv_data()
    plot_histogram(data, label, args.metric)


if __name__ == '__main__':
    main()

