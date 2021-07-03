#ifndef MLP_LOSS_H
#define MLP_LOSS_H

#include <Eigen/Dense>
#include <iostream>
#include <math.h>

using namespace Eigen;
using namespace std;

class LossFunction{
public:
    virtual double loss(VectorXd predicted, VectorXd truth) = 0;
    virtual VectorXd grad(VectorXd predicted, VectorXd truth) = 0;
};

class SoftmaxCrossEntropy : public LossFunction{
public:
    double loss(VectorXd predicted, VectorXd expected){
        double predicted_expected_sum = (predicted.array() * expected.array()).sum();
        return - predicted_expected_sum + log(predicted.array().exp().sum());
    }

    VectorXd grad(VectorXd predicted, VectorXd expected){
        VectorXd predicted_exp = predicted.array().exp();
        VectorXd softmax = predicted_exp / predicted_exp.sum();
        return softmax - expected;
    }
};

#endif //MLP_LOSS_H