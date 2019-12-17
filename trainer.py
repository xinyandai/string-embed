import time
import torch.optim as optim

from tqdm import tqdm
from networks import TripletNet
from networks import TripletLoss
from networks import TwoLayerCNN, MultiLayerCNN


def updateLearningRate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def train_epoch(args, train_loader, device):
    N = len(train_loader)
    C, M = train_loader.C, train_loader.M

    if args.dataset == "word":
        EmbeddingNet = TwoLayerCNN
    else:
        EmbeddingNet = MultiLayerCNN
    net = EmbeddingNet(C, M, embedding=args.embed_dim).to(device)
    model = TripletNet(net).to(device)
    losser = TripletLoss()
    model.train()
    momentum = 0.9
    lrs = {0: 0.001, 10: 0.001, 5: 0.0001, 15: 0.0001}
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=momentum, weight_decay=5e-4
    )

    for epoch in range(args.epochs):
        if epoch in lrs:
            updateLearningRate(optimizer, lrs[epoch] / args.batch_size)
        agg = 0.0
        optimizer.zero_grad()
        with tqdm(range(N), desc="# training") as p_bar:
            start_time = time.time()
            for idx in p_bar:
                x, dists = train_loader[idx]
                x = (i.to(device) for i in x)

                output = model(x)
                loss = losser(output, dists, epoch)
                loss.backward()

                if idx % args.batch_size == 0 or idx == N - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                agg += loss.item()
                p_bar.set_description(
                    str({"Train Epoch": epoch, "Loss": agg / float(idx + 1)})
                )

            print(
                {
                    "# Train Epoch": epoch,
                    "Loss": agg / float(N),
                    "Time": time.time() - start_time,
                }
            )

    return model
