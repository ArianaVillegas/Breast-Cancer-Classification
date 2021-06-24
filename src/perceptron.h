#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <time.h>
#include <math.h>
#include <vector>
#include <numeric>

#include "utils.h"

class Perceptron {
private:
    int size;
    VECTOR weights;
    double bias, output, accum;

    double calculate_net(VECTOR input){
        return bias + inner_product(this->weights.begin(), this->weights.end(), input.begin(), 0);
    }

    double sigmoid(double val){
        return 1.0/(1.0 + exp(-val));
    }

    double sigmoid_derivative(double val){
        return val * (1.0 - val);
    }

public:
    Perceptron(int size){
        this->size = size;
        weights.resize(size);
        for (size_t i = 0; i < size; ++i) {
            weights[i] = (double)rand()/RAND_MAX;
        }
        bias = (double)rand()/RAND_MAX;
    }

    double calculate_output(VECTOR input){
        output = sigmoid(calculate_net(input));
        return output;
    }

    double get_output(){ return output; }

    void set_accum(double error){
        accum = error * sigmoid_derivative(output);
    }

    double get_weight_accum(int idx){
        return weights[idx] * accum;
    }

    void add_weight(double weight, int idx) { weights[idx] += weight * accum; }

    void add_bias(double alpha) { bias += alpha * accum; }

    ~Perceptron(){}
};

#endif
