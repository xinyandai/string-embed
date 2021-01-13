#include <algorithm>

#include <cstdio>
#include <cstdlib>



#include "utils.h"
#include "progress_bar.h"



using namespace std;


void k_medoids( int threshold, int n_medoids, int iter,
                const vector<string >& base_strings,
                const vector<string >& base_modified,  
                const vector<string >& query_strings,
                const vector<string >& query_modified) {
  int nb = base_strings.size();
  std::cout << "reservoir sampling random medoids \n";
  vector<int > medoids = reservoir_sample(nb, n_medoids);
  vector<vector<int > > clusters(n_medoids, vector<int >());

  vector<int > assigned(nb, 0);
  vector<int > distortion(nb, -1); 
  // intialization
  std::cout << "intialization distortion\n";
#pragma omp parallel for
  for (int bi = 0; bi < nb; ++bi) {
    distortion[bi] = edit_distance(base_strings[bi], base_strings[medoids[assigned[bi]]]);
  }

  ProgressBar bar(iter, "Lloyd");
  for (int i = 0; i < iter; ++i) {
    ++bar;
    // assignment
#pragma omp parallel for
    for (int bi = 0; bi < nb; ++bi) {
      int assigned_bi = assigned[bi];
      int distort_bi = distortion[bi];
      for (int mi = 0; mi < n_medoids; ++mi) {
        if (mi != assigned_bi) {
          int dist_mi = edit_distance( base_modified[bi].c_str(),
                                       base_strings[bi].size(),
                                       base_strings[medoids[mi]].c_str(),
                                       base_strings[medoids[mi]].size(),
                                       distort_bi);
          if (dist_mi == -2) {
            std::cerr << "error when computing distance between " 
                      << base_strings[bi] << " and  " 
                      << base_strings[medoids[mi]] << "\n";
          } else if (dist_mi != -1 && dist_mi < distort_bi) {
            assigned_bi = mi;
            distort_bi = dist_mi;
          }
        }
      }
      assigned[bi] = assigned_bi;
      distortion[bi] = distort_bi;
    }

    // update cluster
    for (auto& c: clusters) {
      c.clear();
    }
    for (int bi = 0; bi < nb; ++bi) {
      clusters[assigned[bi]].push_back(bi);
    }
    for (auto& c: clusters) {
      if (c.empty()) {
        std::cerr << "emtpy cell\n";
      }
    }

    // update medoids
  #pragma omp parallel for
    for (int mi = 0; mi < n_medoids; ++mi) {
      vector<int >& ids = clusters[mi];
      vector<int > cost(ids.size(), 0);
      for (int ai = 0; ai < ids.size(); ++ai) {
        for (int bi = ai + 1; bi < ids.size(); ++bi) {
          int dist_= edit_distance(base_strings[ids[ai]], 
                                   base_strings[ids[bi]]);
          cost[ai] += dist_;
          cost[bi] += dist_;
        }
      }
      int new_medoid_idx = std::distance(cost.begin(), 
                                         std::min_element(cost.begin(), cost.end()));
      medoids[mi] = ids[new_medoid_idx];
    }
    std::cout << " distortion " << 1.0 * std::accumulate(distortion.begin(), distortion.end(), 0) / nb << "\n";
  }


}

/**
 * \threshold 
 * \base_location: a set of strings in a file
 * \query_location: a set of strings in a file
 * \ground_truth: knn neighbors in {base} for each {query}
 * @return
 */
int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: ./bin base_location query_location threshold n_medoids\n");
    return 0;
  }

  string base_location = argv[1];
  string query_location = argv[2];

  int threshold = atoi(argv[3]);
  int n_medoids = atoi(argv[4]);

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

  k_medoids(threshold, n_medoids, 10, 
            base_strings, base_modified, 
            query_strings, query_modified);

  return 0;
}
