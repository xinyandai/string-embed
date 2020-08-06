import os
import time
import tqdm
import torch

import numpy as np

from utils import test_recall
from trainer import train_epoch
from datasets import TripletString, StringDataset
from transformers import BertTokenizer, BertModel

# bert embedding
bert_choice = "bert-base-uncased"
cache_dir = "bert-cache"
tokenizer = BertTokenizer.from_pretrained(bert_choice, cache_dir=cache_dir)
bert = BertModel.from_pretrained(bert_choice, cache_dir=cache_dir)


def _batch_embed(args, net, vecs: StringDataset, device, char_alphabet=None):
    """
    char_alphabet[dict]: id to char
    """
    # convert it into a raw string dataset
    if char_alphabet != None:
        vecs.to_bert_dataset(char_alphabet)

    test_loader = torch.utils.data.DataLoader(vecs, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    net.eval()
    embedding = []
    with tqdm.tqdm(total=len(test_loader), desc="# batch embedding") as p_bar:
        for i, x in enumerate(test_loader):
            p_bar.update(1)
            if char_alphabet != None:
                for xx in x:
                    xx = tokenizer(xx, return_tensors="pt")
                    # 1 x 768
                    xx = bert(**xx)[0][0][1].unsqueeze(0)
                    embedding.append(xx.cpu().data.numpy())
            else:

                embedding.append(net(x.to(device)).cpu().data.numpy())
    vecs.to_original_dataset()
    return np.concatenate(embedding, axis=0)


def cnn_embedding(args, h, data_file):
    """
    h[DataHandler]
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_loader = TripletString(h.xt, h.nt, h.train_knn, h.train_dist, K=args.k)

    model_file = "{}/model.torch".format(data_file)
    if os.path.isfile(model_file):
        model = torch.load(model_file)
    else:
        start_time = time.time()
        model = train_epoch(args, train_loader, device)
        if args.save_model:
            torch.save(model, model_file)
        train_time = time.time() - start_time
        print("# Training time: " + str(train_time))
    model.eval()

    # check if we use bert here
    char_alphabet = None
    if args.bert:
        char_alphabet = h.alphabet

    xt = _batch_embed(args, model.embedding_net, h.xt, device, char_alphabet=char_alphabet)
    start_time = time.time()
    xt = []
    xb = _batch_embed(args, model.embedding_net, h.xb, device, char_alphabet=char_alphabet)
    embed_time = time.time() - start_time
    xq = _batch_embed(args, model.embedding_net, h.xq, device, char_alphabet=char_alphabet)
    print("# Embedding time: " + str(embed_time))
    if args.save_embed:
        if args.embed_dir != "":
            args.embed_dir = args.embed_dir + "/"
        os.makedirs("{}/{}".format(data_file, args.embed_dir), exist_ok=True)
        np.save("{}/{}embedding_xb".format(data_file, args.embed_dir), xb)
        np.save("{}/{}embedding_xt".format(data_file, args.embed_dir), xt)
        np.save("{}/{}embedding_xq".format(data_file, args.embed_dir), xq)

    if args.recall:
        test_recall(xb, xq, h.query_knn)
    return xq, xb, xt
