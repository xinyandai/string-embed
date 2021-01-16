#include <algorithm>

#include <cstdio>
#include <cstdlib>
#include <unordered_map>


#include "utils.h"
#include "progress_bar.h"



using namespace std;



/**
 * \base_location: a set of strings in a file
 * \join_location: ground_truth of knn neighbors in {base} for each {query}
 * @return
 */
int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: ./graph_bin base_location join_location join_location_id threshold\n");
    return 0;
  }

  string base_location = argv[1];
  string join_location = argv[2];
  string join_location_id = argv[3];
  int threshold = atoi(argv[4]);

  size_type num_dict = 0;
  size_type nb = 0;
  size_type nq = 0;

  vector<string > base_strings;
  vector<size_type > signatures(256, 1024);

  cout << "loading base data from " << base_location  << endl;
  load_data(base_location, base_strings, signatures, num_dict, nb);
  cout << "loaded base data, num dict " << num_dict << " nb: " << nb << endl;

  vector<string > base_modified = base_strings;
  for (int j = 0; j < base_modified.size(); j++){
    for(int k = 0;k < 8; k++) base_modified[j].push_back(j>>(8*k));
  }

  unordered_map<string, int> str2ids;
  for (int i = 0; i < base_strings.size(); ++i) {
    str2ids[base_strings[i]] = i;
  }

  cout << "loading joined result from  " << join_location  << endl;
  {
    ifstream str_reader(join_location);
    

    vector<int > ids_1;
    vector<int > ids_2;
    

    string line;
    while (getline(str_reader, line)) {
      getline(str_reader, line);
      if (str2ids.find(line) == str2ids.end()) {
        std::cerr << "failed to find str\n"; 
        exit(1);
      }
      ids_1.push_back(str2ids[line]);
      getline(str_reader, line);
      if (str2ids.find(line) == str2ids.end()) {
        std::cerr << "failed to find str\n"; 
        exit(1);
      }
      ids_2.push_back(str2ids[line]);
      getline(str_reader, line);
    }
    str_reader.close();

    cout << "evaluating distances\n";

    vector<int > dists(ids_1.size(), -1);
    ProgressBar bar(ids_1.size(), "evaluating dists");
#pragma omp parallel for
    for (int i = 0; i < ids_1.size(); ++i) {
#pragma omp critical
        {
          ++bar;
        }
        dists[i] = edit_distance( base_modified[ids_1[i]].c_str(),
                                  base_strings[ids_1[i]].size(),
                                  base_modified[ids_2[i]].c_str(),
                                  base_strings[ids_2[i]].size(),
                                  threshold);
    }

    cout << "saving result\n";
    ofstream str_writer(join_location_id);
    for (int i = 0; i < ids_1.size(); ++i) {
        str_writer << ids_1[i] << "\t" << ids_2[i] << "\t" << dists[i] << "\n";
    } 
    str_writer.close();
  }
  
  return 0;
}
