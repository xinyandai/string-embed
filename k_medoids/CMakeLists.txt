cmake_minimum_required (VERSION 2.6)
project (string-cluster)

include_directories("${PROJECT_BINARY_DIR}")



set(SOURCE_EXE k_medoids.cc)            
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
add_executable(k_medoids k_medoids.cc)
add_executable(convert_tool convert_tool.cc)
add_executable(graph graph.cc)