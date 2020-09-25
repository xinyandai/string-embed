# [Convolutional Embedding for Edit Distance (SIGIR 20)](https://arxiv.org/abs/2001.11692)

In this project, we design and implement a deep learning model, which
transforms strings into real number vectors  while  preserving  their 
neighboring relation.  Specifically,  if  the  edit  distance  of two 
strings x and y is small,  the L2-distance of their embeddings should 
also  be  small.  With  this  model,  we can transform expensive edit 
distance  computation to cheaper L2-distance computation and speed up
string similarity search. 

### start training

- train CNN-ED model
```    
python main.py --dataset word --nt 1000 --nq 1000 --epochs 20 --save-split --recall
```

- test bert embedding
```
python main.py --dataset word --nt 1000 --nq 1000 --bert --save-split --recall
```

##### optional arguments:
      -h, --help            show this help message and exit
      --dataset             dataset name which is under folder ./data/
      --nt                  # of training samples
      --nr                  # of generated training samples
      --nq                  # of query items
      --nb                  # of base items
      --k                   # sampling threshold
      --epochs              # of epochs
      --shuffle-seed        seed for shuffle
      --batch-size          batch size for sgd
      --test-batch-size     batch size for test
      --channel CHANNEL     # of channels
      --embed-dim           output dimension
      --save-model          save cnn model
      --save-split          save split data folder
      --save-embed          save embedding
      --random-train        generate random training samples and replace
      --random-append-train generate random training samples and append
      --embed-dir           embedding save location
      --recall              print recall
      --embed EMBED         embedding method
      --maxl MAXL           max length of strings
      --no-cuda             disables GPU training
