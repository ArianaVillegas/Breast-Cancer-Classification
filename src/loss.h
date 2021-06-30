#ifndef MLP_LOSS_H
#define MLP_LOSS_H

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

VectorXd softmax(VectorXd input){
    auto val = (input.array() - input.maxCoeff()).array().exp();
    return val / val.sum();
}

class CrossEntropy {
public:
    double loss(VectorXd predicted, VectorXd truth){
        return -(truth.array() * softmax(predicted).array().log().array()).mean();
    }
};

#endif //MLP_LOSS_H