#include <iostream>
#include <fstream>
#include <sstream>
#include "mlp.h"

using namespace std;

int main() {

    srand(time(0));

    ifstream fin("dataset.csv");
    MATRIX dataset;
    VECTOR labels;
    string line, attr, temp;
    while(!fin.eof()){
        fin >> line;
        stringstream  s(line);
        getline(s, attr, ',');
        getline(s, attr, ',');
        if(attr == "1") labels.push_back(1.0);
        else labels.push_back(0.0);
        VECTOR row;
        while(getline(s, attr, ',')){
            row.push_back(stod(attr));
        }
        dataset.push_back(row);
    }

    /*MATRIX dataset = {{2.7810836,2.550537003},
                      {1.465489372,2.362125076},
                      {3.396561688,4.400293529},
                      {1.38807019,1.850220317},
                      {3.06407232,3.005305973},
                      {7.627531214,2.759262235},
                      {5.332441248,2.088626775},
                      {6.922596716,1.77106367},
                      {8.675418651,-0.242068655},
                      {7.673756466,3.508563011}};
    VECTOR labels = {0,0,0,0,0,1,1,1,1,1};*/

    int n_inputs = dataset[0].size();
    int n_outputs = 2;

    MLP mlp(1, n_inputs, n_outputs);
    mlp.train(dataset, labels, 0.5, 50, n_outputs, true);
    /*VECTOR output = mlp.predict(dataset);
    int cnt = 0;
    for (int i = 0; i < output.size(); +debug+i) {
        if (labels[i] == output[i]) cnt++;
        // cout << "Truth: " << labels[i] << "    Predicted: " << output[i] << '\n';
    }
    cout << "Accuracy: " << cnt / output.size();*/

    return 0;
}
