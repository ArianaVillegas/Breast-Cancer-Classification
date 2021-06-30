#ifndef MLP_MLP2_H
#define MLP_MLP2_H

#include "layer.h"
#include "loss.h"

class MLP {
private:
    int n_input, n_output, size;
    VECTOR n_hidden;
    VECTOR error_report;
    vector<Layer*> mlp;
    CrossEntropy cross_entropy;

    VectorXd propagate(VectorXd input){
        for (Layer* layer:mlp) {
            input = layer->calculate_output(input);
        }
        return input;
    }

    void back_propagate(VectorXd y_truth){
        // Output layer
        Layer *layer = mlp[size-1];
        layer->set_accum(y_truth);

        // Hidden layers
        for (int i = size-2; i >= 0; --i) {
            layer = mlp[i];
            VectorXd y_truth = mlp[i+1]->get_weight_accum();
            layer->set_accum(y_truth);
        }
    }

    void update_weights(VectorXd x, double alpha){
        for (int i = 0; i < size; ++i) {
            Layer *layer = mlp[i];

            // Update weights
            layer->update_weights(x, alpha);

            // Fill new vector
            if (i != size) {
                x = layer->get_output();
            }
        }
    }

public:
    MLP(int n_input, int n_output, VECTOR n_hidden, string activation){
        this->n_input = n_input;
        this->n_output = n_output;
        this->n_hidden = n_hidden;
        this->size = n_hidden.size()+1;

        n_hidden.insert(n_hidden.begin(), n_input);
        n_hidden.push_back(n_output);
        for (int i = 0; i < this->size; ++i) {
            Layer *l = new Layer(n_hidden[i], n_hidden[i+1], activation);
            mlp.push_back(l);
        }
    }

    void train(MATRIX dataset, VECTOR labels, double alpha, int epochs, int n_outputs, bool debug=false){
        int fact = epochs/100;
        MatrixXd _dataset = min_max_scaler(to_eigen_matrix(dataset));
        // cout << _dataset << '\n';
        for (int i = 0; i < epochs; ++i) {
            double error = 0.0;
            for (int j = 0; j < _dataset.rows(); ++j) {
                VectorXd input = _dataset.row(j);
                VectorXd output = propagate(input);
                VectorXd expected = VectorXd::Zero(n_outputs);
                expected[labels[j]] = 1.0;
                error += cross_entropy.loss(output, expected);
                back_propagate(expected);
                update_weights(input, alpha);
            }
            error_report.push_back(error);
            if(debug && (i+1)%fact == 0) cout << "Epoch " << i+1 << " Error = " << error << '\n';
        }
    }

    VECTOR predict(MATRIX dataset){
        VECTOR result;
        MatrixXd _dataset = min_max_scaler(to_eigen_matrix(dataset));
        for (int i = 0; i < _dataset.rows(); ++i) {
            VectorXd input = _dataset.row(i);
            VectorXd out = propagate(input);
            VECTOR output(out.data(), out.data() + out.size());
            result.push_back(max_element(output.begin(), output.end()) - output.begin());
        }
        return result;
    }

    VECTOR get_error_report(){
        return error_report;
    }

    ~MLP(){}
};

#endif
