#ifndef MLP_H
#define MLP_H

#include <algorithm>
#include "layer.h"

using namespace std;

class MLP {
private:
    int n_layer, n_hidden, n_output;
    vector<Layer*> mlp;

    VECTOR propagate(VECTOR input){
        VECTOR new_input;
        for (Layer* layer:mlp) {
            new_input.clear();
            for (Perceptron* perceptron:layer->get_layer()) {
                new_input.push_back(perceptron->calculate_output(input));
            }
            input = new_input;
        }
        return input;
    }

    void back_propagate(VECTOR y_truth){
        // Output layer
        auto layer = mlp[n_layer]->get_layer();
        for (int i = 0; i < n_output; ++i) {
            Perceptron* perceptron = layer[i];
            double error = y_truth[i] - perceptron->get_output();
            perceptron->set_accum(error);
        }

        // Hidden layers
        for (int i = n_layer-1; i >= 0; --i) {
            layer = mlp[i]->get_layer();
            for (int j = 0; j <= n_layer; ++j) {
                double error = 0.0;
                for (Perceptron* perceptron:mlp[i + 1]->get_layer()) {
                    error += perceptron->get_weight_accum(j);
                }
                layer[j]->set_accum(error);
            }
        }
    }

    void update_weights(VECTOR x, double alpha){
        for (int i = 0; i <= n_layer; ++i) {
            auto layer = mlp[i]->get_layer();

            // Update weights
            for (Perceptron* p:layer) {
                for (int j = 0; j < x.size(); ++j) {
                    p->add_weight(alpha * x[j], j);
                }
                p->add_bias(alpha);
            }

            // Fill new vector
            VECTOR new_x;
            for (int j = 0; j < n_hidden; ++j) {
                new_x.push_back(layer[j]->get_output());
            }
            x = new_x;
        }
    }

public:
    MLP(int n_layer, int n_hidden, int n_output){
        this->n_layer = n_layer;
        this->n_output = n_output;
        this->n_hidden = n_hidden;
        for (int i = 0; i < n_layer; ++i) {
            mlp.push_back(new Layer(n_hidden));
        }
        mlp.push_back(new Layer(n_output));
    }

    void train(MATRIX dataset, VECTOR labels, double alpha, int epochs, int n_outputs, bool debug=false){
        for (int i = 0; i < epochs; ++i) {
            double error = 0.0;
            for (int j = 0; j < dataset.size(); ++j) {
                VECTOR output = propagate(dataset[j]);
                VECTOR expected(n_outputs, 0.0);
                expected[labels[j]] = 1;
                error += mse(output, expected);
                back_propagate(expected);
                update_weights(dataset[j], alpha);
            }
            if(debug) cout << "Epoch " << i << " Error = " << error << '\n';
        }
    }

    VECTOR predict(MATRIX dataset){
        VECTOR result;
        for (int i = 0; i < dataset.size(); ++i) {
            VECTOR output = propagate(dataset[i]);
            result.push_back(max_element(output.begin(), output.end()) - output.begin());
        }
        return result;
    }

    ~MLP(){}
};

#endif
