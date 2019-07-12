from random import shuffle

num_query = 1000
num_base = 100000
num_train = 10000


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
    lines = readlines("word/word")
    shuffle(lines)
    query = open("word/query.txt", "w")
    train = open("word/train.txt", "w")
    base = open("word/base.txt", "w")
    print(lines)
    writelines(query, lines[:num_query])
    writelines(train, lines[num_query: num_query + num_train])
    writelines(base, lines[num_query + num_train : num_query + num_train + num_base])
    query.close()
    train.close()
    base.close()

main()
