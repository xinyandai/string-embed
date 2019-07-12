////////////////////////////////////////////////////////////////////////////////
/// Copyright 2018-present Xinyan DAI<xinyan.dai@outlook.com>
///
/// permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions ofthe Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.

/// @version 0.1
/// @author  Xinyan DAI
/// @contact xinyan.dai@outlook.com
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include <assert.h>
#include <cstring>
#include <functional>
#include <fstream>
#ifdef _OPENMP
#   include <omp.h>
#endif
#include <sstream>
#include <vector>

#include <boost/progress.hpp>

#include "heap.hpp"
#include "edlib/edlib.h"

namespace ss {

    using std::vector;
    using std::cout;
    using std::endl;
    using std::ios;
    using std::ifstream;
    using std::ofstream;
    using std::string;

    namespace GroundWriter {

        template<typename DataType>
        void WriteLSHBOX(const char *bench_file, const vector<vector<MaxHeapElement<int> > > &knn) {
            // lshbox file
            ofstream lshboxFout(bench_file);
            if (!lshboxFout) {
                cout << "cannot create output file " << bench_file << endl;
                assert(false);
            }
            int K = knn[0].size();
            lshboxFout << knn.size() << "\t" << K << endl;
            for (int i = 0; i < knn.size(); ++i) {
                assert(knn[i].size() == K);
                lshboxFout << i << "\t";
                const vector<MaxHeapElement<int>>& topker = knn[i];
                for (int idx = 0; idx < topker.size(); ++idx) {
                    lshboxFout << topker[idx].data() << "\t" << topker[idx].dist() << "\t";
                }
                lshboxFout << endl;
            }
            lshboxFout.close();
            cout << "lshbox groundtruth are written into " << bench_file << endl;
        }

        template<typename DataType, typename DistType>
        void WriteIVECS(const char *bench_file, const vector<vector<MaxHeapElement<int, DistType> > > &knn) {
            // ivecs file
            ofstream fout(bench_file, ios::binary);
            if (!fout) {
                cout << "cannot create output file " << bench_file << endl;
                assert(false);
            }
            int K = knn[0].size();
            for (int i = 0; i < knn.size(); ++i) {
                assert(knn[i].size() == K);
                fout.write((char*)&K, sizeof(int));
                const vector<MaxHeapElement<int, DistType>> topker = knn[i];
                for (int idx = 0; idx < topker.size(); ++idx) {
                    fout.write((char*)&topker[idx].data(), sizeof(int));
                }
            }
            fout.close();
            cout << "ivecs groundtruth are written into " << bench_file << endl;
        }

        vector<vector<MaxHeapElement<int> > > ReadLSHBOX(const char* bench_file) {

            int size;
            int k;

            string line;
            ifstream reader(bench_file);

            if (!reader) {
                std::cerr << "can not open file " << bench_file << std::endl;
                assert(false);
            }

            getline(reader, line);
            std::istringstream line_stream(line);
            line_stream >> size >> k;

            vector<vector<MaxHeapElement<int> > > knns(size, vector<MaxHeapElement<int >>());

            for (int i = 0; i < size; ++i) {
                getline(reader, line);
                std::istringstream line_stream(line);

                int id;
                line_stream >> id;

                knns[id].reserve(k);

                for (int j = 0; j < k; ++j) {
                    int distance;
                    int neighbor;
                    line_stream >> neighbor >> distance ;
                    knns[id].emplace_back(MaxHeapElement<int >(distance, neighbor));
                }
            }
            return knns;
        }
    } // end namespace GroundWriter


    template<typename DataType >
    vector<vector<MaxHeapElement<int> > > ExactKNN(
            const DataType *    queries,
            const int           num_queries,
            const DataType *    items,
            const int           num_items,
            const int           dim,
            int                 K,
            std::function<float(const DataType *, const DataType *, const int )> dist) {

        using HeapType = Heap<int, DataType, MaxHeapElement<int, DataType>> ;

        vector<HeapType> heaps(num_queries, HeapType(K));
        vector<vector<MaxHeapElement<int>>> knns(num_queries);

        boost::progress_display progress(num_queries);

#pragma omp parallel for
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {

#pragma omp critical
            {
                    ++progress;
            }

            for (size_t item_id = 0; item_id < num_items; ++item_id) {
                heaps[query_id].Insert(dist(&queries[query_id * dim], &items[item_id * dim], dim), item_id);
            }
            knns[query_id] = heaps[query_id].GetTopK();
	}

        return knns;
    }

    template <typename DataType>
    vector<vector<MaxHeapElement<int, int> > > StringKNN(
            char **       queries,
            int           num_queries,
            char **       items,
            int           num_items,
            int           K,
            std::function<DataType(const char *, const char *)> dist) {

        using HeapType = Heap<int, DataType, MaxHeapElement<int, DataType>> ;

        vector<HeapType> heaps(num_queries, HeapType(K));
        vector<vector<MaxHeapElement<int, int>>> knns(num_queries);

        boost::progress_display progress(num_queries);

#pragma omp parallel for
        for (size_t query_id = 0; query_id < num_queries; ++query_id) {

#pragma omp critical
            {
                ++progress;
            }

            for (size_t item_id = 0; item_id < num_items; ++item_id) {
                heaps[query_id].Insert(dist(queries[query_id], items[item_id]), item_id);
            }

            knns[query_id] = heaps[query_id].GetTopK();
        }
        return knns;
}

    /* A function wrapper for edit distance calculation. */
    int editDistFunc(const char *s, const char *t){
        return edlibAlign(s, strlen(s), t, strlen(t),
            edlibDefaultAlignConfig()).editDistance;
    }

    template <typename DataType>
    std::function<DataType(const char *, const char *)> editDist() {
        std::function<DataType(const char *, const char *)> result = editDistFunc;
        return result;
    }
}