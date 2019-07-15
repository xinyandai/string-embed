# String Embedding

In this project, we design and implement a deep learning model, which
transforms strings into real number vectors  while  preserving  their 
neighboring relation.  Specifically,  if  the  edit  distance  of two 
strings x and y is small,  the L2-distance of their embeddings should 
also  be  small.  With  this  model,  we can transform expensive edit 
distance  computation to cheaper L2-distance computation and speed up
string similarity search. 

### sample train/base/query data from "data/word/word"

    cd data
    python generate.py
    tree 
    .
    ├── generate.py
    └── word
        ├── base.txt
        ├── query.txt
        ├── train.txt
        └── word

### compute ground truth and knn

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


### start training

    pyhton main.py


