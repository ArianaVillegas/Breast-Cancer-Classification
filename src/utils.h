#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

#define VECTOR vector<double>
#define MATRIX vector<VECTOR>

pair<MATRIX, VECTOR> readCSV(string filename){
    ifstream fin(filename);
    MATRIX dataset;
    VECTOR labels;
    string line, attr, temp;
    while(!fin.eof()){
        fin >> line;
        stringstream  s(line);
        getline(s, attr, ',');
        getline(s, attr, ',');
        if(attr == "B") labels.push_back(1.0);
        else labels.push_back(0.0);
        VECTOR row;
        while(getline(s, attr, ',')){
            row.push_back(stod(attr));
        }
        dataset.push_back(row);
    }
    return {dataset, labels};
}

MatrixXd to_eigen_matrix(MATRIX m){
    int rows = m.size(), cols = m[0].size();
    MatrixXd _m(rows, cols+1);
    for (int i = 0; i < rows; ++i) {
        m[i].push_back(0);
        _m.row(i) = Map<VectorXd, Unaligned>(m[i].data(), m[i].size());
    }
    return _m;
}
