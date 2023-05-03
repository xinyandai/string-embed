# [Convolutional Embedding for Edit Distance (SIGIR 20)](https://arxiv.org/abs/2001.11692)

Original model - In this project, we design and implement a deep learning model, which
transforms strings into real number vectors  while  preserving  their 
neighboring relation.  Specifically,  if  the  edit  distance  of two 
strings x and y is small,  the L2-distance of their embeddings should 
also  be  small.  With  this  model,  we can transform expensive edit 
distance  computation to cheaper L2-distance computation and speed up
string similarity search. 

Updated model proposed by Kobie Arndt, Nathan Ballay and William Kressner - In the modified version
of this project, we added the ability to invoke a new loss function known as LeakyReLU.  The 
user can specify the value for the hyper parameter used.  Our intentions for this proposal 
is to address the "dying ReLU" problem in hopes to achieve an improved item-recall.  The project
is ran the same as below.  Note: commands were run using python3 instead of python.



### before run
Please install PyTorch refer to [PyTorch](https://pytorch.org/get-started/locally/) 
and install Levenshtein and transformers via 
```
pip install python-Levenshtein
pip install transformers
```


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
      --leaky_relu          # modify loss function with specified hyper parameter
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



# reference
If you use this code, please cite the following [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401045)
```
@inproceedings{cnned,
  author    = {Xinyan Dai and
               Xiao Yan and
               Kaiwen Zhou and
               Yuxuan Wang and
               Han Yang and
               James Cheng},
  title     = {Convolutional Embedding for Edit Distance},
  booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} conference on
               research and development in Information Retrieval, {SIGIR} 2020, Virtual
               Event, China, July 25-30, 2020},
  pages     = {599--608},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3397271.3401045},
  doi       = {10.1145/3397271.3401045},
}
```

