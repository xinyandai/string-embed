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

#include <algorithm>
#include <utility>
#include <vector>

using std::vector;
using std::pair;

template<typename DataT, typename DistType=float>
class DistanceElement {
protected:
    DistType    _dist;
    DataT       _data;

public:

    DistanceElement(DistType dist, const DataT& data): _dist(dist), _data(data) {}

    explicit DistanceElement(const pair<DistType, DataT>& p): _dist(p.first), _data(p.second) {}


    DistType dist() const {
        return _dist;
    }

    const DataT& data() const {
        return _data;
    }
};

// min heap element
template<typename DataT, typename DistType=float>
class MinHeapElement : public DistanceElement<DataT> {
public:

    MinHeapElement(DistType dist, const DataT& data) : DistanceElement<DataT>(dist, data) {}

    bool operator<(const MinHeapElement& other) const  {
        return this->_dist > other._dist;
    }
};

/// max heap element
template<typename DataT, typename DistType=float>
class MaxHeapElement : public DistanceElement<DataT, DistType> {
public:

    MaxHeapElement(DistType dist, const DataT& data) : DistanceElement<DataT, DistType>(dist, data) {}

    bool operator<(const MaxHeapElement& other) const  {
        return this->_dist < other._dist;
    }
};

/// TODO(Xinyan) : use std::vector and std::make_heap to replace priority queue
template <
        typename DataT,
        typename DistType=float,
        typename HeapElement=MaxHeapElement<DataT, DistType>
>
class Heap {

private:
    int                  _K;
    vector<HeapElement > _heap;
public:
    explicit Heap(int K) {
        _K = K;
    }

    bool Insert(const HeapElement& pair) {
        if (_heap.size() < _K) {
            _heap.push_back(pair);
            std::push_heap(_heap.begin(), _heap.end());
            return true;
        } else {
            if (pair.dist() < _heap[0].dist()) {
                std::pop_heap(_heap.begin(), _heap.end());
                _heap[_K-1] =  pair;/// pop the max one and swap it
                std::push_heap(_heap.begin(), _heap.end());
                return true;
            }
        }
        return false;
    }


    bool Insert(DistType dist, DataT data) {
        return Insert(HeapElement(dist, data));
    }


    vector<HeapElement> GetTopK() const {
        vector<HeapElement > heap = _heap;
        std::sort(heap.begin(), heap.end());
        return heap;
    }

    /**
     * generate TopK.
     */
    vector<pair<float, int > > TopKPairs() const {
        vector<HeapElement > heap = GetTopK();
        vector<pair<float, int > > topK(heap.size());
        for (int i = 0; i < heap.size(); ++i) {
            topK[i] = std::make_pair(heap[i].dist(), heap[i].data());
        }
        return topK;
    }

    int getK() const {
        return _K;
    }

};
