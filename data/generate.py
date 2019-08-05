from random import shuffle

# num_query = 1000
# num_base = 100000
# num_train = 10000
# dataset = "word"

# total = 1385451
# num_query = 1000
# num_base = 100000
# num_train = 10000
# dataset = "dblp"

# total = 1134188
# num_query = 1000
# num_base = 200000
# num_train = 10000
# dataset = "querylog"

max_length = 59419
total = 245567
num_query = 1000
num_base = 20000
num_train = 10000
dataset = "enron"

# total = 347949
# num_query = 1000
# num_base = 200000
# num_train = 10000
# dataset = "trec"

def readlines(file):
    """
    :param file: the path to the file
    :return: a list of string
    """
    lines = open(file, 'rb').read().splitlines()
    return [line.decode("utf8", "ignore") for line in lines]


def writelines(writer, lines):
    writer.write("\n".join(lines))

def main():
    lines = readlines("%s/%s" % (dataset, dataset))
    shuffle(lines)
    query = open("%s/query.txt" % dataset, "w")
    train = open("%s/train.txt" % dataset, "w")
    base = open("%s/base.txt" % dataset, "w")

    writelines(query, lines[:num_query])
    writelines(train, lines[num_query: num_query + num_train])
    writelines(base, lines[num_query + num_train :
                           num_query + num_train + num_base])
    query.close()
    train.close()
    base.close()

main()
