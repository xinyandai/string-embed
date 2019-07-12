#pragma once

#include <climits>
#include <cstring>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <climits>
#include <fstream>

using               std::vector;
using               std::map;
using               std::pair;
using               std::string;

class StringReader {
protected:
    int             line;
    int             size;
    char**          data;

public:
    StringReader();

    char**          getData();
    int             getSize();
    int             getLine();

    char**          read(const char* filename, int lineLim = INT_MAX, int sizeLim = INT_MAX);
    void            free();
};

class SegmentReader: public StringReader {
private:
    vector<int>                 length;
    vector<int>                 start;
    vector<pair<int, int> >     segmentTable;
    int                         segSize;
    int                         segNum;

public:
    SegmentReader(int _segSize);
    int                         getSegNum();
    int                         getSegSize();
    int                         getLength(int i);
    int                         getStart(int i);
    char**                      read(const char* filename, int lineLim = INT_MAX, int sizeLim = INT_MAX);
    void                        free();
};

class IVECSReader{
private:
    int                     line;
    string                  filename;
    vector<vector<int> >    data;

public:
    vector<vector<int> >    read(string filename, int line);
    vector<vector<int> >    getData();
};

StringReader::StringReader(): line(0), size(0) {
    data = NULL;
}

char** StringReader::getData() { return data; }
int    StringReader::getSize() { return size; }
int    StringReader::getLine() { return line; }

char** StringReader::read(const char* filename,
    int lineLim, int sizeLim) {
    vector<string> vec;
    std::ifstream f(filename);
    std::string buf;
    while(std::getline(f, buf)){
        if(!(line < lineLim && size < sizeLim)) break;
        vec.push_back(buf);
        size += buf.size();
        line++;
    }

    data = new char* [line];
    for(int i = 0; i < line; i++){
        data[i] = new char [vec[i].size() + 1];
        strcpy(data[i], vec[i].c_str());
    }

    f.close();
    return data;
}

void StringReader::free() {
    for(int i = 0; i < line; i++) { delete[] data[i]; }
    delete[] data;
}

SegmentReader::SegmentReader(int _segSize): StringReader(), segSize(_segSize) {
    segNum = 0;
}

int SegmentReader::getSegNum(){
    return segNum;
}

int SegmentReader::getSegSize(){
    return segSize;
}

int SegmentReader::getLength(int i) {
    return length[i];
}

int SegmentReader::getStart(int i){
    return start[i];
}

char** SegmentReader::read(const char* filename, int lineLim, int sizeLim) {
    vector<string> vec;
    std::ifstream f(filename);
    std::string buf;
    while(std::getline(f, buf)){
        if(!(line < lineLim && size < sizeLim)) break;
        vec.push_back(buf);
        size += buf.size();
        line++;
    }

    vector<string> segs;
    for(int i = 0; i < line; i++) {
        length.push_back(vec[i].length());
        start.push_back(segs.size());
        for(int j = 0; j * segSize < length[i]; j++) {
            segs.push_back(vec[i].substr(j * segSize, std::min(segSize, length[i] - j * segSize)));
            segmentTable.push_back(std::make_pair(i, j));
            segNum++;
        }
    }

    data = new char* [segNum];
    for(int i = 0; i < segNum; i++){
        data[i] = new char [segs[i].size() + 1];
        strcpy(data[i], segs[i].c_str());
    }

    return data;
}

void SegmentReader::free() {
    for(int i = 0; i < segNum; i++) { delete[] data[i]; }
    delete[] data;
}

vector<vector<int> > IVECSReader::read(string filename, int line){
    data.clear();

    int dim;
    int lineCount = 0;
    std::ifstream f(filename, std::ios::in | std::ios::binary);
    while(lineCount < line){
        f.read((char *) (&dim), 4);
        int num;
        vector<int> v;
        for(int i = 0; i < dim; i++) {
            f.read((char *) (&num), 4);
            v.push_back(num);
        }
        data.push_back(v);
        lineCount++;
    }
    return data;
}

vector<vector<int> > IVECSReader::getData(){
    return data;
}