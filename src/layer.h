#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <time.h>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "utils.h"
#include "activation.h"
#include "optimizer.h"

using namespace std;
using namespace Eigen;


class Layer {
private:
    int in_size, out_size;
    MatrixXd weights;
    VectorXd bias;
    VectorXd output;
    VectorXd accum;
    ActivationFunction* activation;
    OptimizerFunction* optimizer;

    void select_activation_function(string name){
        if(name == "sigmoid") activation = new Sigmoid();
        else if(name == "tanh") activation = new Tanh();
        else if(name == "relu") activation = new RELU();
        else activation = new NoActivation();
    }

    void select_optimizer_function(string name){
        if(name == "adam") optimizer = new Adam(in_size, out_size);
        else optimizer = new NoOptimizer();
    }

public:
    Layer(int in_size, int out_size, string activation, string optimizer){
        this->in_size = in_size;
        this->out_size = out_size;
        weights = (MatrixXd::Random(in_size, out_size).array() + 1)/2;
        bias = (VectorXd::Random(out_size).array() + 1)/2;
        output = VectorXd::Zero(out_size);
        accum = VectorXd::Zero(out_size);
        select_activation_function(activation);
        select_optimizer_function(optimizer);
    }

    VectorXd calculate_output(VectorXd input){
        output = activation->calculate((input.transpose() * weights).transpose() + bias);
        return output;
    }

    void set_accum(VectorXd loss){
        accum = (loss.array() * activation->calculate_derivative(output).array());
    }

    VectorXd get_weight_accum(){
        return accum.transpose()*weights.transpose();
    }

    void update_weights(VectorXd x, double alpha){
        MatrixXd x_m(x.size(),1), accum_m(1,accum.size());
        x_m.col(0) = x;
        accum_m.row(0) = accum;

        weights = optimizer->calculate_w(weights, x_m*accum_m, alpha);
        bias = optimizer->calculate_b(bias, accum, alpha);
    }

    VectorXd get_output(){
        return output;
    }
};

#endif
