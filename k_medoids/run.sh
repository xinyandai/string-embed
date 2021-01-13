mkdir  -p build
cd build
cmake .. && make -j
./k_medoids  ../../folder/word_0/808/base ../../folder/word_0/808/query 5 1024 
