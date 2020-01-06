import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fontsize = 44
ticksize = 40
labelsize = 35
legendsize = 30
plt.style.use("seaborn-white")

W = 12.0
H = 9.5
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

    plt.ylabel("Accuracy",
               fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.legend(
        loc="lower right",
        fontsize=legendsize
    )
    plt.gcf().set_size_inches(W, H)
    plt.subplots_adjust(
        top=0.976,
        bottom=0.141,
        left=0.133,
        right=0.988,
        hspace=0.2,
        wspace=0.2
    )
    # plt.show()



def plot_one(top, location, label, color, linestyle, marker=""):
    data = from_text(location)
    y = data[:, top]
    x = data[:, items]
    plt.plot(x / 1000.0, y, color, label=label, linestyle=linestyle, marker=marker)


def plot_time():
    path = "logs/nt1000nq1000_time"
    tops = [1, 5, 10, 20, 50, 100, 1000]
    for dataset in ["uniref", "dblp", "trec", "gen50ks", "enron"]:
        for top in range(3, 10):
            plt.clf()

            plot_one(top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "blue", "--", "*")
            plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "red", "-.", "s")
            plot_one(
                top, "{}/{}/{}".format(path, "gru", dataset), "GRU", "black", "-.", "^"
            )
            _plot_setting()
            plt.xlabel("# Probe Time [S]", fontsize=fontsize)
            plt.xscale("log")
            ax = plt.subplot()
            plt.text(
                x=0.75, y=0.5, color='blue',
                s=dataset.upper(),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.text(
                x=0.75, y=0.4, color='blue',
                s="TOP-{}".format(tops[top - 2]),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.show()
            # plt.savefig("/home/xinyan/Dropbox/project/xinyan/string-embedding/figures/top{}_{}.pdf".format(tops[top-2], dataset))

def plot_items():
    path = "logs/nt1000nq1000"
    tops = [1, 5, 10, 20, 50, 100, 1000]
    for dataset in ["uniref", "dblp", "trec", "gen50ks", "enron"]:
        for top in range(2, 9):
            plt.clf()

            plot_one(top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "blue", "--", "*")
            plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "red", "-.", "s")
            plot_one(
                top, "{}/{}/{}".format(path, "gru", dataset), "GRU", "black", "-.", "^"
            )
            _plot_setting()
            plt.xscale("linear")
            plt.xlabel("# Probe Items [K]", fontsize=fontsize)
            ax = plt.subplot()
            plt.text(
                x=0.75, y=0.5, color='blue',
                s=dataset.upper(),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.text(
                x=0.75, y=0.4, color='blue',
                s="TOP-{}".format(tops[top - 2]),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.show()
            # plt.savefig("/home/xinyan/Dropbox/project/xinyan/string-embedding/figures/top{}_{}.pdf".format(tops[top-2], dataset))

plot_time()

