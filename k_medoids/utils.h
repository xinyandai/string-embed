//
// Created by xinyan on 30/12/2019.
//

#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include <numeric>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <chrono>
#include <ctime>
#include <time.h>

#include "verification.h"


using std::string;
using std::ifstream;
using std::ofstream;
using std::endl;
using std::cout;
using std::iota;
using std::min;
using std::pair;
using std::vector;
using size_type = unsigned;

template< class IntType = int >
vector<IntType> reservoir_sample(const IntType& N, const IntType& k) {
  vector<IntType> sample;
  if (k == 0) { 
    return sample;
  }
  std::default_random_engine gen;
  IntType i;
  for (i = 0; i != k; ++i) {
    sample.push_back(i);
  }
  for (; i < N; ++i) {
    std::uniform_int_distribution<IntType > dist_place(0, i);
    IntType j = dist_place(gen);
    if (j < k)
      sample[j] = i;
  }
  return sample;
}

void load_data( const string& str_location,
                vector<string>& strings,
                vector<size_type >& signatures,
                size_type& num_dict, size_type& num_str) {

  ifstream  str_reader(str_location);

  num_str = 0;

  string line;
  while (getline(str_reader, line)) {
    // record the string
    strings.push_back(line);
    // record the number of strings
    num_str++;
    // record the signatures and the number of identical characters

    for (char c : line) {
      if (signatures[c] == 1024) {
        signatures[c] = num_dict++;
      }
    }
  }
  str_reader.close();

}



/** A timer object measures elapsed time,
 * and it is very similar to boost::timer. */
class timer {
 public:
  timer() { restart(); }
  ~timer() = default;
  /** Restart the timer. */
  void restart() {
    t_start = std::chrono::high_resolution_clock::now();
  }
  /** @return The elapsed time */
  double elapsed() {
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point t_start;
};