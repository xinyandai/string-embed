import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fontsize = 44
ticksize = 40
legendsize = 30
plt.style.use("seaborn-white")
plot_accuracy = True
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
    plt.xlabel("# Probe Items", fontsize=fontsize)
    plt.ylabel("Accuracy" if plot_accuracy else "Loss", fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.xscale("log")
    plt.legend(
        loc="lower right" if plot_accuracy else "upper right", fontsize=legendsize
    )

    matplotlib.pyplot.gcf().set_size_inches(W, H)
    matplotlib.pyplot.subplots_adjust(
        top=0.984, bottom=0.141, left=0.133, right=0.988, hspace=0.2, wspace=0.2
    )
    # plt.show()



def plot_one(top, location, label, color, linestyle, marker=""):
    data = from_text(location)
    y = data[:, top]
    x = data[:, items]
    plt.plot(x, y, color, label=label, linestyle=linestyle, marker=marker)


def plot():
    path = "logs/nt1000nq1000"
    tops = [1, 5, 10, 20, 50, 100, 1000]
    for dataset in ["enron", "trec", "gen50ks"]:
        for top in range(2, 9):
            plt.clf()
            # plot_one(top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "blue", "--", "*")
            plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "red", "-.", "s")
            plot_one(
                top, "{}/{}/{}".format(path, "gru", dataset), "GRU", "black", "-.", "^"
            )
            _plot_setting()
            plt.savefig("/home/xinyan/Dropbox/project/xinyan/string-embedding/figures/top{}_{}.pdf".format(tops[top-2], dataset))


plot()

