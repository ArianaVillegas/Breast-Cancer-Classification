#include <vector>
#include <iostream>

using namespace std;

#define VECTOR vector<double>
#define MATRIX vector<VECTOR>

double mse(VECTOR a, VECTOR b){
    double error = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        error += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return error;
}