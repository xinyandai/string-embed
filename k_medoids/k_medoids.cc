#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <functional>


#include "utils.h"
#include "progress_bar.h"



using namespace std;

class HierarchicalQuantization {
  private:
    int depth_; 
    int parent_;
    int max_nodes_;
    int max_child_;
    int n_medoids_; 
    vector<int > items_;
    vector<int > distort_;
    vector<int > medoids_;
    vector<HierarchicalQuantization > children_;
    std::function<int(int i, int j, int t)> distor_base_;
    std::function<int(int i, int j, int t)> distor_query_;

    /**
     * @pre_condition items_ is not empty
     * @post_condition medoids_, children_ 
     **/
    void init() {
      n_medoids_  = std::min<int >((items_.size() + max_nodes_ - 1) / max_nodes_, max_child_);
      if (is_leaf()) {
        return;
      }
      medoids_.reserve(n_medoids_);
      vector<int > indices = reservoir_sample<int >(items_.size(), n_medoids_);
      for (int i = 0; i < n_medoids_; ++i) {
        medoids_.emplace_back(items_[ indices[i] ]);
      }

      children_.reserve(n_medoids_);
      for (int mi = 0; mi < n_medoids_; ++mi) {
        children_.emplace_back(depth_+1, medoids_[mi], max_nodes_, max_child_, distor_base_, distor_query_);
      }
    }
    
    void associate() {
      if (is_leaf()) {
        return;
      }
#pragma omp parallel for
      for (int i = 0; i < items_.size(); i++) {
        int cm = 0;
        int dm = distor_base_(items_[i], medoids_[cm], -1);
        for (int ci = 1; ci < n_medoids_; ++ci) {
          int di = distor_base_(items_[i], medoids_[ci], dm);
          if (di < dm) {
            cm = ci;
            dm = di;
          }
          if (di == 0) {
            break;
          }
        }
#pragma omp critical
        {
          children_[cm].add_item(items_[i], dm);
        }
      }
#pragma omp parallel for
      for (int k = 0; k < children_.size(); ++k) {
        children_[k].run();
      }
    }

  public:
   /**
    * every thing is initilized empty and HierarchicalQuantization is leaf node by default
    * 1. call set_items/add_item
    * 2. call run to init medoids and assoicate clusters
    */
    HierarchicalQuantization( int depth, int parent, int max_nodes, int max_child,  
                              const std::function<int(int i, int j, int t)>& distor_base, 
                              const std::function<int(int i, int j, int t)>& distor_query)
                            : depth_(depth), parent_(parent), max_nodes_(max_nodes), max_child_(max_child), 
                              n_medoids_(-1), items_(), distort_(), medoids_(), 
                              distor_base_(distor_base), distor_query_(distor_query) {
    }
    
    void run() {
      init();
      associate();
    }

    int query(int qid, int T) {
      if (is_leaf()) {
        int exact_dist = distor_query_(qid, parent_, -1);
        if (exact_dist <= T)
          return 0;

        int pruned = 0;
        for (int i = 0; i < items_.size(); i++) {
          if (exact_dist - distort_[i] > T) {
            pruned++;
          }
        }
        return pruned;
      } else {
        int pruned = 0;
        for (auto& c : children_) {
          pruned += c.query(qid, T);
        }
        return pruned;
      }
    }
    void set_items(const vector<int >& items, const vector<int >& distorts) {
      items_ = items;
      distort_ = distorts;
    }
    void add_item(int id, int distort) {
      items_.push_back(id);
      distort_.push_back(distort);
    }
    bool is_leaf() const {
      if (n_medoids_ == -1) {
        throw std::runtime_error("HQ is not initilized!");
      }
      if (n_medoids_ == 0) {
        std::cerr << "parent_ : " << parent_ << "\n"
                  << "size i_ : " << items_.size() << "\n"
                  << "at level: " << depth_ << "\n"; 
        throw std::runtime_error("HQ with 0 medoids");
      }
      return n_medoids_ == 1;
    }

};

void k_medoids( int threshold, int n_medoids, int iter,
                const vector<string >& base_strings,
                const vector<string >& base_modified,  
                const vector<string >& query_strings,
                const vector<string >& query_modified) {

  std::function<int(int i, int j, int t)> distor_base = [&](int i, int j, int t) {
    if (i == j) {
      return 0;
    }
    t = t == -1? base_strings[i].size() + base_strings[j].size() : t;
    int d = edit_distance( base_modified[i].c_str(),
                           base_strings[i].size(),
                           base_modified[j].c_str(),
                           base_strings[j].size(),
                           t);
    return d < 0 ? t + 1 : d;
  };
  std::function<int(int i, int j, int t)> distor_query = [&](int i, int j, int t) {
    t = t == -1? base_strings[i].size() + base_strings[j].size() : t;
    int d = edit_distance( query_modified[i].c_str(),
                           query_strings[i].size(),
                           base_modified[j].c_str(),
                           base_strings[j].size(),
                           t);
    return d < 0 ? t + 1 : d;
  };
  const int nb = base_strings.size();
  const int nq = query_strings.size();
  const int max_child = 16; 
  const int max_nodes = 16 * max_child; // there are {16}-{max_nodes} items in each of cluster
  const int approx_n_centers = nb / max_nodes ; 

  std::cout << "# number of items " << nb << "\n"
            << "# approximate number of leaf medoids " << approx_n_centers << "\n"
            << "# maximal nodes in each leaf cluster " << max_nodes << "\n";
  HierarchicalQuantization hq(1, 0, max_nodes, max_child, distor_base, distor_query);
  
  vector<int > idx(nb);
  const vector<int > dists(nb, PRUNE_K);
  std::iota(idx.begin(), idx.end(), 0);
  hq.set_items(idx, dists);
  hq.run();
  vector<int > pruned(nq, 0);

  vector<int > ts = {1, 10, 40};
  for (int T : ts) {
#pragma omp parallel for
    for (int qi = 0; qi < nq; ++qi) {
      pruned[qi] = hq.query(qi, T);
    }
    
    long sum_pruned = 0;
    for (int p : pruned) {
      sum_pruned += p;
    }
    std::cout << "average number of pruned rate when T=" << T 
              << ", pruned="  << 1.0 * sum_pruned / nq / nb << "\n"; 
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
