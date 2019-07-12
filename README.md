## String Embedding
# prepare data
- sample train/base/query data from "data/word/word"


    cd data/word
    python generate.py
    tree 
    .
    ├── generate.py
    └── word
        ├── base.txt
        ├── query.txt
        ├── train.txt
        └── word

- compute ground truth and knn


    cd ground
    make 
    ./ground ../data/word/query.txt 1000 ../data/word/base.txt 100000 1000 ../data/word/gt.ivecs 80
    ./ground ../data/word/train.txt  10000 ../data/word/train.txt  10000 10000 ../data/word/knn.ivecs 80
    
    tree ../data
    ../data
    ├── generate.py
    └── word
        ├── base.txt
        ├── gt.ivecs
        ├── knn.ivecs
        ├── query.txt
        ├── train.txt
        └── word


- we have the following files



