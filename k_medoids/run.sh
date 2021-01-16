mkdir  -p build
cd build
cmake .. && make -j
data="trec"
./k_medoids  ../../folder/${data}_0/808/base ../../folder/${data}_0/808/query 5 1024 
