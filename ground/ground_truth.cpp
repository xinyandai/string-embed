#include <iostream>
#include <cstring>
#include <cstdlib>
#include <climits>
#include "utils/ground_truth.hpp"
#include "utils/reader.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

using std::string;
typedef vector<vector<MaxHeapElement<int, int> > > KNN;

int main(int argc, char *argv[]){
    /*  Input format:
    *  <EXECUTABLE> <QUERY_FILE> <QUERY_NUMBER> <DATA_FILE> <DATA_NUMBER> <K> <OUTPUT> <NUMTHREAD = 1>
    */

    if(!(argc == 7 || argc == 8)) {
        std::cout << ">> Invalid input!" << "\n";
        std::cout << ">> Standard input: <EXECUTABLE> <QUERY_FILE> <QUERY_NUMBER> <DATA_FILE> <DATA_NUMBER> <K> <OUTPUT> <NUMTHREAD = 1>." << "\n";
        exit(0);
    }

    int numThread = 1;
    if(argc == 8) numThread = atoi(argv[7]);

#ifdef _OPENMP
    omp_set_num_threads(numThread);
#endif

    string queryName("data/");
    queryName += argv[1];

    string dataName("data/");
    dataName += argv[3];

    StringReader queryReader;
    char** query = queryReader.read(queryName.c_str(), atoi(argv[2]), INT_MAX);

    StringReader dataReader;
    char** data = dataReader.read(dataName.c_str(), atoi(argv[4]), INT_MAX);

    std::cout << ">> Start sorting...\n";
    KNN knn = ss::StringKNN <int>(query, queryReader.getLine(),
                                    data,  dataReader.getLine(),
                                    atoi(argv[5]), ss::editDist<double> ());

    string outputFile("data/ground/");
    outputFile += argv[6];
    ss::GroundWriter::WriteIVECS<int, int> (outputFile.c_str(), knn);
    std::cout << ">> Finish finding ground truth! Output file: " << outputFile << "\n";
    return 0;
}
