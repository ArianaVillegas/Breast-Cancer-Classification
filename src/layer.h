#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <time.h>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "utils.h"
#include "activation.h"

using namespace std;
using namespace Eigen;


class Layer {
private:
    MatrixXd weights;
    VectorXd bias;
    VectorXd output;
    VectorXd accum;
    ActivationFunction* function;

    void select_activation_function(string name){
        if(name == "sigmoid") function = new Sigmoid();
        else if(name == "tanh") function = new Tanh();
        else function = new RELU();
    }

public:
    Layer(int in_size, int out_size, string activation){
        weights = MatrixXd::Random(in_size, out_size);
        bias = VectorXd::Random(out_size);
        output = VectorXd::Zero(out_size);
        accum = VectorXd::Zero(out_size);
        select_activation_function(activation);
    }

    VectorXd calculate_output(VectorXd input){
        output = function->calculate((input.transpose() * weights).transpose() + bias);
        return output;
    }

    void set_accum(VectorXd y_truth){
        VectorXd error = y_truth - output;
        accum = (error.array() * function->calculate_derivative(output).array());
    }

    VectorXd get_weight_accum(){
        return accum.transpose()*weights.transpose();
    }

    void update_weights(VectorXd x, double alpha){
        MatrixXd x_m(x.size(),1), accum_m(1,accum.size());
        x_m.col(0) = x;
        accum_m.row(0) = accum;
        weights += alpha * x_m * accum_m;
        bias += alpha * accum;
    }

    VectorXd get_output(){
        return output;
    }
};

#endif
