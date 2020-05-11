import time

from tqdm import tqdm
from networks import *


def train_epoch(args, train_set, device):
    N = len(train_set)
    C, M = train_set.C, train_set.M

    torch.manual_seed(time.time())

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    if args.dataset == "word":
        EmbeddingNet = TwoLayerCNN
    elif args.dataset == "querylog":
        EmbeddingNet = QuerylogCNN
    elif args.dataset == "enron":
        EmbeddingNet = EnronCNN
    elif args.dataset == "trec":
        EmbeddingNet = TrecCNN
    elif args.dataset == "dblp":
        EmbeddingNet = DBLPCNN
    elif args.dataset == "uniref":
        EmbeddingNet = UnirefCNN
    else:
        EmbeddingNet = MultiLayerCNN

    if args.epochs == 0 and args.dataset != "word":
        EmbeddingNet = RandomCNN
    
    net = EmbeddingNet(C, M, embedding=args.embed_dim, channel=args.channel, mtc_input=args.mtc).to(device)
    model = TripletNet(net).to(device)
    losser = TripletLoss(args)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with tqdm(total=args.epochs * len(train_loader), desc="# training") as p_bar:
        for epoch in range(args.epochs):
            agg = 0.0
            agg_r = 0
            agg_m = 0
            optimizer.zero_grad()

            start_time = time.time()
            for idx, batch in enumerate(train_loader):
                (
                    anchor, pos, neg,
                    anchor_len, pos_len, neg_len,
                    pos_dist, neg_dist, pos_neg_dist,
                ) = (i.to(device) for i in batch)

                optimizer.zero_grad()
                output = model((anchor, pos, neg))

                r, m, loss = losser(
                    output,
                    (anchor_len, pos_len, neg_len),
                    (pos_dist, neg_dist, pos_neg_dist),
                    epoch,
                )

                loss.backward()
                optimizer.step()

                agg += loss.item()
                agg_r += r.item()
                agg_m += m.item()
                p_bar.update(1)
                p_bar.set_description(
                    "# Epoch: %3d Time: %.3f Loss: %.3f  r: %.3f m: %.3f"
                    % (
                        epoch,
                        time.time() - start_time,
                        agg / (idx + 1),
                        agg_r / (idx + 1),
                        agg_m / (idx + 1),
                    )
                )
    return model
