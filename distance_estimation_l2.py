import numpy as np
from utils import l2_dist
from nns import linear_fit, load_vec, get_args



threshold = 1000

args = get_args()

# args.embed = 'gru'
# xq_gru, xb_gru, xt_gru, train_dist, query_dist = load_vec(args)
# query_dist = query_dist[:, :50000]
# xb_gru = xb_gru[:50000, :]
#
# train_idx = np.where(train_dist < threshold)
# query_idx = np.where(query_dist < threshold)
#
# train_dist_l2_gru = l2_dist(xt_gru, xt_gru)
#
# l2ed_gru = linear_fit(
#     train_dist_l2_gru[train_idx],
#     train_dist[train_idx], deg=1)
#
# query_dist_l2_gru = l2_dist(xq_gru, xb_gru)
# print(np.mean(np.abs(l2ed_gru(query_dist_l2_gru[query_idx]) / query_dist[query_idx] - 1.0)))


# args.embed = 'cnn'
# xq_cnn, xb_cnn, xt_cnn,train_dist, query_dist = load_vec(args)
# query_dist = query_dist[:, :50000]
# xb_cnn = xb_cnn[:50000, :]
#
# train_idx = np.where(train_dist < threshold)
# query_idx = np.where(query_dist < threshold)
#
# print("# training all pair distance")
# train_dist_l2_cnn = l2_dist(xt_cnn, xt_cnn)
# print("# training all pair distance fitting to edit distance")
# l2ed_cnn = linear_fit(
#     train_dist_l2_cnn[train_idx],
#     train_dist[train_idx],
#     deg=1)
# print("# query all pair distance")
# query_dist_l2_cnn = l2_dist(xq_cnn, xb_cnn)
# print("# fitting errors")
# print(np.mean(np.abs(l2ed_cnn(query_dist_l2_cnn)[query_idx] / query_dist[query_idx] - 1.0)))


import matplotlib.pyplot as plt
fontsize = 44
ticksize = 40
labelsize = 35
legendsize = 30
plt.style.use("seaborn-white")

W = 12.0
H = 9.5
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
print("# plotting")
# idx = np.random.choice(np.size(query_dist[query_idx]), threshold)

# plt.scatter(query_dist[query_idx].reshape(-1)[idx],
#             l2ed_gru(query_dist_l2_gru[query_idx].reshape(-1))[idx], color="blue")

# plt.scatter(query_dist[query_idx].reshape(-1)[idx],
#             l2ed_cnn(query_dist_l2_cnn[query_idx].reshape(-1))[idx], color="red")
# plt.scatter(query_dist[query_idx].reshape(-1)[idx],
#             query_dist[query_idx].reshape(-1)[idx], color="black")

plt.xlim(left=-10, right=threshold)
plt.ylim(bottom=-10, top=threshold)
_plot_setting()
plt.xlabel("True Edit Distance", fontsize=fontsize)
plt.ylabel("Estimated Edit Distance", fontsize=fontsize)
plt.text(
    x=0.02, y=0.8, color='blue',
    s=args.dataset.upper(),
    fontsize=labelsize,
    transform=plt.subplot().transAxes
)
# plt.savefig("/home/xinyan/Dropbox/project/Yan Xiao Paper/string-embedding/figures/distance_estimation_{}.pdf".format(args.dataset))
plt.show()