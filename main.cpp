#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include "src/split.h"
#include "src/bootstrap.h"
using namespace std;

int main() {
    Eigen::initParallel();

    srand(time(0));

    auto[dataset, labels] = readCSV("dataset.csv");

    SPLIT s(labels, dataset, 0.7);

    int n_inputs = dataset[0].size();
    int n_outputs = 2;
    VECTOR n_hidden = {15, 15};


    MLP mlp(n_inputs, n_outputs, n_hidden, "relu", "adam");
    mlp.train(s.x_train, s.y_train, 0.001, 1000, n_outputs, true);


    VECTOR output = mlp.predict(s.x_test);
    int cnt = 0;
    for (int i = 0; i < output.size(); ++i) {
        if (s.y_test[i] == output[i]) cnt++;
    }
    cout << "Size: " << output.size() << "  Accuracy: " << cnt*1.0 / output.size() * 100 << "%\n";

    return 0;

}
