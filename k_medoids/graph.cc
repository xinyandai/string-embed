#include <algorithm>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <queue> 

#include "utils.h"
#include "progress_bar.h"

using namespace std;

void 
load_joined(string location, vector<vector<pair<int, int > > >& nns) {
    ifstream fs(location.c_str());
    if (!fs) {
        std::cout << "cannot open file " << location << std::endl;
        exit(1);
    }
    string line;
    while(getline(fs, line)) {
        istringstream iss(line);
        int i, j, d;
        iss >> i >> j >> d;
        nns[i].emplace_back(d, j);
        nns[j].emplace_back(d, i);
    }
}

std::tuple<int, int, int> 
pruned_search(  int threshold, int join_threshold, int pre_evaluated,
                const vector<string >& base_strings,
                const vector<string >& base_modified,  
                const string& query_string,
                const string& query_modified, 
                const vector<vector<pair<int, int > > >& nns) {
  int nb = base_strings.size();

  std::random_device rd;  
  std::mt19937 gen(rd()); 
  vector<size_t> idx(nb);
  iota(idx.begin(), idx.end(), 0);
  // std::shuffle(idx.begin(), idx.end(), gen);
  stable_sort(idx.begin(), idx.end(),
       [&nns](size_t i1, size_t i2) {return nns[i1].size() > nns[i2].size();});
  int idx_i = 0;
  
   
   // d = -1 default 
   // d = -2 traigle pruned
   // d = -3 traigle keeped
   // threshold >= d >= 0  evaluated and keep
   // d =  join_threshold + threshold evaluated
  vector<int > dists(nb, -1);
  int evaluated = 0;
  int pruned = 0;
  int keeped = 0;
  int processed = 0;
  for (idx_i = 0; idx_i < pre_evaluated; ++idx_i) {
    int i = idx[idx_i];
    if (dists[i] != -1) {
      continue;
    }
    evaluated++;
    int dist_i = edit_distance( base_modified[i].c_str(),
                                base_strings[i].size(),
                                query_modified.c_str(),
                                query_string.size(),
                                threshold + join_threshold);

    if (dist_i >= 0) {
      dists[i] = dist_i;
      for (const pair<int, int >& nbs : nns[i]) {
        if (dists[nbs.second] == -1) {
          if (dist_i + nbs.first <= threshold) {
              dists[nbs.second] = -3; // trangle keeped
              keeped++;
          }
          else if (dist_i - nbs.first > threshold) {
              dists[nbs.second] = -2; // trangle pruned
              pruned++;
          }

          processed++;
        }
      }
    } else {
      dists[i] = threshold + join_threshold + 1;
      for (const pair<int, int >& nbs : nns[i]) {
        if (dists[nbs.second] == -1) {

          dists[nbs.second] = -2;
          pruned++;
          processed++;
        }
      }
    }
  }
  return std::make_tuple(pruned, keeped, processed);
}

/**
 * \threshold 
 * \base_location: a set of strings in a file
 * \query_location: a set of strings in a file
 * \join_location: knn neighbors in {base} for each {query}
 * @return
 */
int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: ./graph_bin base_location query_location threshold join_threshold join_location\n");
    return 0;
  }

  string base_location = argv[1];
  string query_location = argv[2];

  int threshold = atoi(argv[3]);
  int join_threshold = atoi(argv[4]);
  string join_location = argv[5];

  size_type num_dict = 0;
  size_type nb = 0;
  size_type nq = 0;

  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);

  cout << "loading base  data " << base_location  << endl;
  cout << "loading query data " << query_location << endl;
  load_data(base_location, base_strings, signatures, num_dict, nb);
  cout << "loaded base data, num dict " << num_dict << " nb: " << nb << endl;
  load_data(query_location, query_strings, signatures, num_dict, nq);
  cout << "loaded query data, num dict " << num_dict << " nq: " << nq << endl;


  vector<string > base_modified = base_strings;
  for (int j = 0; j < base_modified.size(); j++){
    for(int k = 0;k < 8; k++) base_modified[j].push_back(j>>(8*k));
  }
  vector<string > query_modified = query_strings;
  for (int j = 0; j < query_modified.size(); j++){
    for(int k = 0;k < 8; k++) query_modified[j].push_back(j>>(8*k));
  }

  cout << "loading joined result\n";
  vector<vector<pair<int, int > > > nns(nb, vector<pair<int, int > >());
  load_joined(join_location, nns);

  vector<int > n_keeped(nq, 0);
  vector<int > n_pruned(nq, 0);
  vector<int > n_precessed(nq, 0);

  int pre_evaluated = nb / 4;
  cout << "searching with pre_evaluated=" << pre_evaluated << "\n";
#pragma omp parallel for
  for (int i = 0; i < nq; ++i) {
    std::tuple<int, int, int> res = pruned_search( threshold, join_threshold, pre_evaluated, 
                                             base_strings, base_modified, 
                                             query_strings[i], query_modified[i], nns);
    n_pruned[i] = std::get<0>(res);
    n_keeped[i] = std::get<1>(res);
    n_precessed[i] = std::get<2>(res);
  }

  long sum_pruned = 0;
  long sum_keeped = 0;
  long sum_processed = 0;
  for (int i = 0; i < nq; i++) {
      sum_keeped += n_keeped[i];
      sum_pruned += n_pruned[i];
      sum_processed += n_precessed[i];
  }

  std::cout << "keeped rate " << 1.0 * sum_keeped / nq / nb << "\n";
  std::cout << "pruned rate " << 1.0 * sum_pruned / nq / nb << "\n";
  std::cout << "process rate " << 1.0 * sum_pruned / nq / nb << "\n";
  return 0;
}
