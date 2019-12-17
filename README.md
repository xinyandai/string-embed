# String Embedding

In this project, we design and implement a deep learning model, which
transforms strings into real number vectors  while  preserving  their 
neighboring relation.  Specifically,  if  the  edit  distance  of two 
strings x and y is small,  the L2-distance of their embeddings should 
also  be  small.  With  this  model,  we can transform expensive edit 
distance  computation to cheaper L2-distance computation and speed up
string similarity search. 

### start training

    python main.py


