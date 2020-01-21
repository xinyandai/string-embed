import numpy as np
import matplotlib
import matplotlib.pyplot as plt


fontsize = 44
ticksize = 40
labelsize = 38
legendsize = 35
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
    return np.genfromtxt(location)


def _plot_setting():
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
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
    plt.plot(x / 1000.0, y, color, label=label, linestyle=linestyle, marker=marker, markersize=12, linewidth=2.5)


def plot_threshold():
    path = "logs/nt1000nq1000_threshold"
    def precision_recall(location):
        rp = from_text(location)
        assert len(rp) % 2 == 0, location + "precision and recall does not match"
        return rp[:len(rp)//2, :], rp[len(rp)//2:, :]
    for dataset in ["uniref"]:
        cnn_recall, cnn_precision = precision_recall("{}/{}/{}".format(path, "cnn", dataset))
        gru_recall, gru_precision = precision_recall("{}/{}/{}".format(path, "gru", dataset))
        assert len(cnn_recall) == len(gru_recall)

        print(cnn_precision)
        for i in range(len(cnn_recall)):
            plt.clf()
            threshold = cnn_recall[i, 0]
            plt.plot(cnn_recall[i, 2:], cnn_precision[i, 2:], color="r", label="CNN", linestyle="-.", marker="s",)
            plt.plot(gru_recall[i, 2:], gru_precision[i, 2:], color="k", label="GRU", linestyle="-.", marker="^",)
            _plot_setting()
            plt.xlabel("Recall",    fontsize=fontsize)
            plt.ylabel("Precision", fontsize=fontsize)
            plt.xscale("linear")
            plt.legend(
                loc="lower left",
                fontsize=legendsize
            )
            plt.xlim(right=1.01)
            plt.xlim(left=0.85)
            ax = plt.subplot()
            plt.text(
                x=0.06, y=0.35, color='blue',
                s=dataset.upper(),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.text(
                x=0.06, y=0.25, color='blue',
                s="K=%d"%(threshold),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.show()
            plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                        "string-embedding/figures/threshold_%s_%d.pdf"
                        %(dataset, threshold))

def plot_time():
    path = "logs/nt1000nq1000_time"
    tops = [1, 5, 10, 20, 50, 100, 1000]
    for dataset in ["dblp", "uniref", "trec", "gen50ks", "enron"]:
        for top in range(3, 10):
            plt.clf()
            plot_one(top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "b", "--", "*")
            plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "r", "-", "s")
            plot_one(top, "{}/{}/{}".format(path, "gru", dataset), "GRU", "k", "-.", "^")
            _plot_setting()
            plt.legend(
                loc="lower right",
                fontsize=legendsize
            )
            plt.xlabel("# Probe Time [S]", fontsize=fontsize)
            plt.ylabel("Accuracy",  fontsize=fontsize)
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
                s="TOP-{}".format(tops[top - 3]),
                fontsize=labelsize,
                transform=ax.transAxes
            )
            plt.show()
            plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                        "string-embedding/figures/top{}_{}.pdf"
                        .format(tops[top-2], dataset))

def plot_items():
    path = "logs/nt1000nq1000"
    tops = [1, 5, 10, 20, 50, 100, 1000]
    # for dataset in ["dblp", "enron", "trec", "uniref", "gen50ks", ]:
    for dataset in ["gen50ks"]:
        for top in range(2, 9):
            plt.clf()
            plot_one(top, "{}/{}/{}".format(path, "cnn", dataset), "CNN", "r", "-", "s")
            # plot_one(top, "{}/{}/{}".format(path, "rnd", dataset), "RND", "r", "-.", "s")
            plot_one(top, "{}/{}/{}".format(path, "gru", dataset), "GRU", "k", "-.", "^")
            plot_one(top, "{}/{}/{}".format(path, "cgk", dataset), "CGK", "b", "--", "*")
            _plot_setting()
            plt.legend(
                loc="lower right",
                fontsize=legendsize
            )
            plt.xscale("linear")
            if dataset == "gen50ks":
                plt.xlim(left=-1, right=23)
            plt.xlabel("# Items [K]", fontsize=fontsize)
            plt.ylabel("Recall", fontsize=fontsize)
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
            # plt.show()
            plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                        "string-embedding/figures/top{}_{}.pdf"
                        .format(tops[top-2], dataset))

def plot_dim():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/dim_{}".format(path, 8), "8", "gray", "--", "s")
        plot_one(top, "{}/dim_{}".format(path, 16), "16", "pink", "-", "*")
        plot_one(top, "{}/dim_{}".format(path, 32), "32", "black", "-.", "x")
        plot_one(top, "{}/dim_{}".format(path, 64), "64", "blue", "--", "o")
        plot_one(top, "{}/dim_{}".format(path, 128), "128", "red", "-", "s")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        # plt.show()
        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_dims_{}.pdf"
                    .format(tops[top-2]))


def plot_channel():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/channel_{}.txt".format(path, 1),   "1",     "gray", "--", "s")
        plot_one(top, "{}/channel_{}.txt".format(path, 2),   "2",    "blue", "-.", "*")
        plot_one(top, "{}/channel_{}.txt".format(path, 4),   "4",    "black", "-", "x")
        plot_one(top, "{}/channel_{}.txt".format(path, 8), "8", "red", "-", "s")
        plot_one(top, "{}/channel_{}.txt".format(path, 16),  "16",   "blue", "-.", ">")
        plot_one(top, "{}/channel_{}.txt".format(path, 32),   "32",    "pink", "--", "o")

        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        # plt.show()
        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_channel_{}.pdf"
                    .format(tops[top-2]))

def plot_epochs():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/epoch_{}".format(path, 5),    "5     Epoch",     "gray", "--", "s")
        plot_one(top, "{}/epoch_{}".format(path, 10),   "10   Epoch",    "blue", "-.", "*")
        plot_one(top, "{}/epoch_{}".format(path, 20),   "20   Epoch",    "black", "-", "x")
        plot_one(top, "{}/epoch_{}".format(path, 50),   "50   Epoch",    "pink", "--", "o")
        plot_one(top, "{}/epoch_{}".format(path, 100),  "100 Epoch",   "blue", "-.", ">")
        plot_one(top, "{}/epoch_{}".format(path, 200),  "200 Epoch",   "red", "-", "s")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        # plt.show()
        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_epochs_{}.pdf"
                    .format(tops[top-2]))


def plot_convs():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/conv_{}".format(path, 8), "8", "gray", "--", "s")
        # plot_one(top, "{}/conv_{}".format(path, 16), "CNN(layer=16)", "pink", "-", "*")
        plot_one(top, "{}/conv_{}".format(path, 9), "9", "black", "-.", "x")
        plot_one(top, "{}/conv_{}".format(path, 10), "10", "blue", "--", "o")
        plot_one(top, "{}/conv_{}".format(path, 12), "12", "red", "-", "s")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        # plt.show()
        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_conv_{}.pdf"
                    .format(tops[top-2]))

def plot_pool():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/pool_{}.txt".format(path, "mean"), "Average Pooling", "red", "-", "s")
        plot_one(top, "{}/pool_{}.txt".format(path, "max"),  "Maximum Pooling", "black", "--", "x")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)

        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_pool_{}.pdf"
                    .format(tops[top-2]))
        # plt.show()


def plot_embed():
    dataset = "trec"
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        K = tops[top-2]
        plt.clf()

        plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 1),   "#Embed=1",  "gray", "--", "o")
        plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 2),   "#Embed=2",    "red", "-", "s")
        # plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 3),   "#Embed=3",    "black", "-", "x")
        # plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 4),   "#Embed=4",    "pink", "--", "o")
        # plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 6),   "#Embed=6",   "blue", "-.", ">")
        # plot_one(top, "{}/{}/embed_{}.txt".format(path, dataset, 8),   "#Embed=8",   "red", "-", "s")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=129)
        plt.xscale("linear")
        plt.xlabel("# Probe Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        plt.text(
            x=0.05, y=0.7, color='blue',
            s=dataset.upper() + "\nK=" + str(K),
            fontsize=labelsize,
            transform=plt.subplot().transAxes
        )
        plt.subplots_adjust(
            top=0.984,
            bottom=0.141,
            left=0.133,
            right=0.955,
            hspace=0.2,
            wspace=0.2
        )
        plt.show()
        # plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
        #             "string-embedding/figures/varies_embed_{}_{}.pdf"
        #             .format(dataset, K))

def plot_loss():
    path = "logs/aba"
    tops = [1, 5, 10, 20, 50, 100, 1000]

    for top in range(2, 9):
        plt.clf()
        plot_one(top, "{}/loss_{}".format(path, "triplet"),
                 "CNN(Triplet Loss)", "black", "-.", "x")
        plot_one(top, "{}/loss_{}".format(path, "abs"),
                 "CNN(Pairwise Loss)", "blue", "--", "o")
        plot_one(top, "{}/loss_{}".format(path, "comb"),
                 "CNN(Composite Loss)", "red", "-", "s")
        _plot_setting()
        plt.legend(
            loc="lower right",
            fontsize=legendsize
        )
        plt.xlim(left=-3, right=63)
        plt.xscale("linear")
        plt.xlabel("# Items [K]", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        # plt.show()
        plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/"
                    "string-embedding/figures/varies_loss_{}.pdf"
                    .format(tops[top-2]))

def plot_bar():
    import numpy as np
    import matplotlib.pyplot as plt
    dataset = "GEN50KS"
    N = 3
    if dataset.upper() == "TREC":
        K = 40
        train_time = np.array([0, 0, 510.748])
        embed_time = np.array([0, 0, 190.15])
        search_time = np.array([0, 0, 8924.61])
        total_time = np.array([128525.479, 16943.97656, 0])
        train_time, embed_time, search_time, total_time = (i / 1000 for i in
                                                           (train_time, embed_time, search_time, total_time))
        plt.ylabel('Time (1000 s)', fontsize=fontsize)
    elif dataset.upper() == "ENRON":
        K = 40
        train_time = np.array([0, 0, 614.496])
        embed_time = np.array([0, 0, 27.421])
        search_time = np.array([0, 0, 65.717])
        total_time = np.array([399.966, 138.259, 0])
        train_time, embed_time, search_time, total_time = (i for i in
                                                           (train_time, embed_time, search_time, total_time))
        plt.ylabel('Time (s)', fontsize=fontsize)
    elif dataset.upper() == "GEN50KS":
        K = 150
        train_time = np.array([0, 0, 160.087])
        embed_time = np.array([0, 0, 8.613])
        search_time = np.array([0, 0, 48.808])
        total_time = np.array([4540.327, 52.834496, 0])
        train_time, embed_time, search_time, total_time = (i for i in
                                                           (train_time, embed_time, search_time, total_time))
        plt.ylabel('Time (s)', fontsize=fontsize)
    else:
        assert False

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence


    p3 = plt.bar(ind, search_time,  width, )
    p1 = plt.bar(ind, train_time,   width, bottom=search_time)
    p2 = plt.bar(ind, embed_time,   width, bottom=train_time + search_time,)
    p4 = plt.bar(ind, total_time,   width, bottom=train_time + embed_time + search_time,)


    plt.xticks(ind, ('PassJoin', 'EmbedJoin', 'CNNJoin'), fontsize=fontsize)
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
               ('Train Time', 'Embed Time', 'Search Time', 'Total Time'),
               fontsize=legendsize)
    ax = plt.subplot()
    plt.text(
        x=0.75, y=0.4, color='blue',
        s="{}\nK={}".format(dataset.upper() , K),
        fontsize=labelsize,
        transform=ax.transAxes
    )
    _plot_setting()

    plt.show()


def plot_threshold_search():
    dataset = "dblp"
    threshold = 40
    logs = "logs/threshold_search/th{}/{}/".format(threshold, dataset)
    exact_embed = np.genfromtxt(logs+"bf_single_thread.txt")
    hs = np.genfromtxt(logs+"hs.txt")
    pq_embed = np.genfromtxt(logs+"pq_single_thread.txt")

    plt.plot(pq_embed[:, 2],    np.zeros_like(pq_embed[:, 2]) +  hs  / 1000.0,           'k', label='HSsearch', linestyle='--', marker='', markersize=12, linewidth=2.5)
    plt.plot([min(np.min(pq_embed[:, 2]), np.min(exact_embed[:, 2])),
              max(np.max(pq_embed[:, 2]), np.max(exact_embed[:, 2]))],
             [hs / 1000.0,  hs / 1000.0],
             'k', label='HSsearch', linestyle='-', marker='^', markersize=12, linewidth=3)
    plt.plot(pq_embed[:, 2],
             pq_embed[:, 1] / 1000.0, 'b', label='CNN-ED-PQ', linestyle='-.', marker='o', markersize=12,
             linewidth=2.5)
    plt.plot( exact_embed[:, 2], exact_embed[:, 1] / 1000.0,  'r', label='CNN-ED', linestyle='-', marker='s', markersize=12, linewidth=2.5)

    _plot_setting()
    plt.legend(
        loc="center left",
        fontsize=legendsize
    )
    plt.text(
        x=0.05, y=0.7, color='blue',
        s=dataset.upper()+"\nK="+str(threshold),
        fontsize=labelsize,
        transform=plt.subplot().transAxes
    )
    plt.xlabel("Recall", fontsize=fontsize)
    plt.ylabel("Time (s)", fontsize=fontsize)
    plt.show()


def plot_threshold_search_gru():
    dataset = "gen50ks"
    threshold = 100
    logs = "logs/threshold_search/th{}/{}/".format(threshold, dataset)
    cnn_embed = np.genfromtxt(logs+"bf.txt")
    gru_embed = np.genfromtxt(logs+"gru_bf.txt")


    plt.plot( cnn_embed[:, 1] / 1000.0, cnn_embed[:, 2],  'r',
              label='CNN', linestyle='-', marker='s', markersize=12, linewidth=2.5)
    plt.plot( np.sort(gru_embed[:, 1]) / 1000.0, np.sort(gru_embed[:, 2]),  'k',
              label='GRU', linestyle='--', marker='o', markersize=12, linewidth=2.5)

    _plot_setting()
    plt.legend(
        loc="center left",
        fontsize=legendsize
    )
    plt.text(
        x=0.05, y=0.7, color='blue',
        s=dataset.upper()+"\nK="+str(threshold),
        fontsize=labelsize,
        transform=plt.subplot().transAxes
    )
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel("Recall", fontsize=fontsize)
    plt.show()


plot_pool()
