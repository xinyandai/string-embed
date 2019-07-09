import math
import torch
import torch.optim as optim
import numpy as np
from sorter import BatchSorter
from datasets import ivecs_read
from datasets import word2vec
from datasets import TripletString
from networks import CNNEmbedding as EmbeddingNet
from networks import TripletNet
from networks import TripletLoss

def test_recall(X, Q, G):
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]
    recalls = BatchSorter(X, Q, X, G, Ts, metric='euclid', batch_size=200).recall()
    print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        print("{}, {}, {}, {}, {}, {}".format(
            2**i, 0, recall, recall * len(G[0]) / t, 0, t))

def main(epoch):
    train_set = "data/word"
    knn = "data/knn.ivecs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = TripletString(train_set, knn)
    N = len(train_loader)
    D = train_loader.D
    model = TripletNet(EmbeddingNet(D).to(device)).to(device)
    losser = TripletLoss()

    epochs = [0, 5]
    lrs = [0.01, 0.001]
    momentum = 0.9
    model.train()
    for epoch in range(epoch):
        for i_epoch, i_lr in zip(epochs, lrs):
            if epoch == i_epoch:
                optimizer = optim.SGD(model.parameters(), lr=i_lr,
                                      momentum=momentum, weight_decay=5e-4)
        agg = 0.0
        for idx in range(N):
            x = train_loader[idx]
            x = (i.to(device) for i in x)
            optimizer.zero_grad()
            output = model(x)
            loss = losser(output)
            loss.backward()
            optimizer.step()
            agg += loss.item()
            # print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, agg / float(N)))

    def _batch_embed(net, vecs):
        chunk_size = 100
        embedding = np.empty(shape=(len(vecs), net.embedding))
        for i in range(math.ceil(len(vecs) / chunk_size)):
            sub = vecs[i * chunk_size: (i + 1) * chunk_size, :]
            embedding[i * chunk_size: (i + 1) * chunk_size, :] = \
                net(torch.from_numpy(sub).to(device)).cpu().data.numpy()
        return embedding

    item = _batch_embed(model.embedding_net, word2vec(train_set)[0][:10000])
    query = _batch_embed(model.embedding_net, word2vec(train_set)[0][:200])

    gt = ivecs_read(knn)[:200, :50]
    test_recall(item, query, gt)

main(10)
"""
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.02, 1.0, 0, 1
2, 0, 0.0388, 0.97, 0, 2
4, 0, 0.067, 0.8375, 0, 4
8, 0, 0.1075, 0.671875, 0, 8
16, 0, 0.1582, 0.494375, 0, 16
32, 0, 0.2186, 0.3415625, 0, 32
64, 0, 0.29100000000000004, 0.22734375000000004, 0, 64
128, 0, 0.3814, 0.148984375, 0, 128
256, 0, 0.4914, 0.0959765625, 0, 256
512, 0, 0.6151, 0.060068359375, 0, 512
1024, 0, 0.7334, 0.035810546875, 0, 1024
2048, 0, 0.8464, 0.0206640625, 0, 2048
4096, 0, 0.9411, 0.011488037109375, 0, 4096
8192, 0, 0.9939, 0.0060662841796875, 0, 8192
16384, 0, 1.0, 0.0030517578125, 0, 16384
"""
"""
bash-4.2$ python main.py 
Train Epoch: 0 Loss: 0.202764
Train Epoch: 1 Loss: 0.128244
Train Epoch: 2 Loss: 0.114933
Train Epoch: 3 Loss: 0.107376
Train Epoch: 4 Loss: 0.105673
Train Epoch: 5 Loss: 0.085288
Train Epoch: 6 Loss: 0.073000
Train Epoch: 7 Loss: 0.064277
Train Epoch: 8 Loss: 0.058501
Train Epoch: 9 Loss: 0.059352
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
expected items, overall time, avg recall, avg precision, avg error, avg items
1, 0, 0.0197, 0.985, 0, 1
2, 0, 0.0371, 0.9275, 0, 2
4, 0, 0.064, 0.8, 0, 4
8, 0, 0.10619999999999999, 0.66375, 0, 8
16, 0, 0.1595, 0.49843750000000003, 0, 16
32, 0, 0.2447, 0.38234375, 0, 32
64, 0, 0.35619999999999996, 0.27828125, 0, 64
128, 0, 0.4974, 0.194296875, 0, 128
256, 0, 0.6548999999999999, 0.12791015625, 0, 256
512, 0, 0.8095, 0.079052734375, 0, 512
1024, 0, 0.927, 0.045263671875, 0, 1024
2048, 0, 0.9856, 0.0240625, 0, 2048
4096, 0, 0.9990000000000001, 0.01219482421875, 0, 4096
8192, 0, 0.9998999999999999, 0.0061029052734375, 0, 8192
16384, 0, 1.0, 0.0030517578125, 0, 16384
"""