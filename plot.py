import numpy as np
import matplotlib.pyplot as plt


fontsize = 44
ticksize = 40
legendsize = 30
plt.style.use("seaborn-white")
plot_accuracy = True

items = 1
top1 = 2
top5 = 3
top10 = 4
top20 = 5
top50 = 6
top100 = 7
top1000 = 8


def from_text(location):
    return np.genfromtxt(location, skip_header=2)


def _plot_setting():
    plt.xlabel("# Probe Items", fontsize=fontsize)
    plt.ylabel("Accuracy" if plot_accuracy else "Loss", fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.xscale("log")
    plt.legend(
        loc="upper left" if plot_accuracy else "upper right", fontsize=legendsize
    )

    plt.show()


def plot_one(top, location, label, color, linestyle, marker=""):
    data = from_text(location)
    y = data[:, top]
    x = data[:, items]
    plt.plot(x, y, color, label=label, linestyle=linestyle, marker=marker)


def plot():
    path = "logs/nt1000nq100"
    dataset = "uniref.txt"
    for top in range(2, 9):
        plot_one(
            top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "black", "--", "*"
        )
        plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "red", "-.", "s")
        _plot_setting()


plot()
