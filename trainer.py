import tqdm
import torch.optim as optim

from networks import TripletNet
from networks import TripletLoss
from networks import TwoLayerCNN as EmbeddingNet

def train_epoch(args, train_loader, device):
    N = len(train_loader)
    C, M = train_loader.C, train_loader.M

    model = TripletNet(EmbeddingNet(C, M, embedding=args.embed_dim).to(device)).to(device)
    losser = TripletLoss()

    epochs = [0, 4]
    lrs = [0.001, 0.0001]
    momentum = 0.9
    model.train()
    with tqdm.tqdm(total=N * args.epochs) as p_bar:
        for epoch in range(args.epochs):
            for i_epoch, i_lr in zip(epochs, lrs):
                if epoch == i_epoch:
                    optimizer = optim.SGD(model.parameters(), lr=i_lr,
                                          momentum=momentum, weight_decay=5e-4)
            agg = 0.0
            for idx in range(N):
                p_bar.update(1)
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

        return model
