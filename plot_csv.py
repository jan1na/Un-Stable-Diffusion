import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from attack_types import run_names

WANDB_DIR = "./wandb/"

COLORS = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#D879D4', '#AA4499', '#882255',
          '#37001C', '#A26F85']

FID_VALUES = [5.52843996652564, 13.5188010775728, 14.4754941578319, 13.0412494709869, 10.5513145035911,
              15.0473063432149, 14.1113076272244, 10.6831156682874, 4.60950168091205, 6.7844135449148]


def read_csv_data(dir: str) -> [np.ndarray, List[str]]:
    """
    Read csv data into a numpy array and a list.

    :return: data numpy array and label list
    """
    old_pwd = os.getcwd()
    os.chdir(WANDB_DIR + dir)
    files = glob.glob("*")
    files.sort(key=os.path.getmtime)

    data, label = [], []
    for file in files:
        filename = os.fsdecode(file)
        # if filename not in ['synonym-word', 'typo-char', 'naive-char', 'original-control']:
        #    continue
        label.append(filename)
        with open(filename) as f:
            data.append([float(v[1:-1]) for v in f.read().splitlines()[1:]])
    os.chdir(old_pwd)
    return np.asarray(data), label


def plot_histogram(data: np.ndarray, label: List[str], title: str, directory: str, y_max: float):
    """
    Plot data in a histogram and save the plot.

    :param data: data that gets plotted
    :param label: label for the data
    :param title: title for the whole plot
    :param directory: dictionary where to save all histograms
    """
    padding = (np.max(data) - np.min(data)) * 0.1
    x_min = np.min(data) - padding
    x_max = np.max(data) + padding
    bins = np.linspace(np.min(data), np.max(data), num=10, endpoint=False)

    plt.hist(data.T, bins, color=COLORS, histtype='bar', label=label)
    plt.legend(prop={'size': 10})
    plt.title(title)
    file_name = title.replace(" ", "_").lower()
    plt.savefig('./wandb/plots/' + file_name + '.pdf')
    plt.show()
    plt.close()

    fig, axs = plt.subplots(5, 2, figsize=(10, 16))
    y, x = 0, 0
    for i in range(len(label)-1):
        if x == 0 and y == 0:
            d = np.vstack((data[i], data[len(label)-1])).T
            l = [label[i], label[-1]]
            c = [COLORS[i], COLORS[-1]]
        else:
            d = data[i].T
            l = label[i]
            c = COLORS[i]
        axs[y, x].hist(d, bins, color=c, histtype='bar', label=l, rwidth=0.8)
        axs[y, x].legend(prop={'size': 12})
        axs[y, x].set_ylim([0, y_max])
        axs[y, x].set_xlim([x_min, x_max])
        axs[y, x].set_xlabel("Cosine Similarity")
        axs[y, x].set_ylabel("Count")
        if x == 0:
            x += 1
        else:
            y += 1
            x = 0
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('./wandb/plots/' + directory + '.pdf')
    plt.show()
    plt.close()


def plot_chart(data: np.ndarray, label: List[str], title: str):
    """
    Plot mean data in a chart and save the plot.

    :param data: data that gets plotted
    :param label: label for the data
    :param title: title for the whole plot
    """

    plt.figure(constrained_layout=True)

    y_pos = np.arange(len(label))
    means = [np.mean(x) for x in data]

    plt.barh(y_pos, means, align='center', color=COLORS)
    plt.yticks(y_pos, labels=label)
    plt.gca().invert_yaxis()
    plt.xlabel("Cosine Similarity")
    plt.title(title)
    padding = (max(means) - min(means)) * 0.1
    x_min = min(means) - padding
    x_max = max(means) + padding
    plt.xlim([x_min, x_max])

    file_name = title.replace(" ", "_").lower()
    plt.savefig('./wandb/plots/' + file_name + '.pdf')
    plt.show()
    plt.close()


def plot_fid_chart(title: str):
    plt.figure(constrained_layout=True)

    labels = run_names + ["random"]
    y_pos = np.arange(len(labels))
    means = FID_VALUES + [FID_VALUES[0]]

    plt.barh(y_pos, means, align='center', color=COLORS)
    plt.yticks(y_pos, labels=labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Cosine Similarity")
    plt.title(title)
    padding = (max(means) - min(means)) * 0.1
    x_min = min(means) - padding
    x_max = max(means) + padding
    plt.xlim([x_min, x_max])

    file_name = title.replace(" ", "_").lower()
    plt.savefig('./wandb/plots/' + file_name + '.pdf')
    plt.show()
    plt.close()


def main():
    data, label = read_csv_data("csv-files-cosine-sim/")
    plot_histogram(data, label, "Image Cosine Similarity", "image_cosine_sim", 5000)
    plot_chart(data, label, "Mean Image Cosine Similarity")

    data, label = read_csv_data("csv-files-image-text-sim/")
    plot_histogram(data, label, "Image Text Similarity", "image_text_sim", 6000)
    plot_chart(data, label, "Mean Image Text Similarity")

    data, label = read_csv_data("csv-files-image-caption-sim/")
    plot_histogram(data, label, "Image Caption Similarity", "image_caption_sim", 5200)
    plot_chart(data, label, "Mean Image Caption Similarity")

    plot_fid_chart("Clean FID Score")


if __name__ == '__main__':
    main()
